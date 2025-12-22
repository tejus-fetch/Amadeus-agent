import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, EmailStr, Field

from app.api import CLIENT, TransferBookingRequest, TransferSearchRequest
import requests
from app.notification import (
    render_transfer_cancellation_email,
    render_transfer_confirmation_email,
    send_email,
)

# ============================================================
# Logging Configuration
# ============================================================

logger = logging.getLogger("amadeus.transfers")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# -------------------------------
# 1. Custom Agent State
# -------------------------------


@dataclass
class UserSessionContext:
    thread_id: str
    agent_id: str


class Location(BaseModel):
    """
    Schema for a location, either an airport or an address.

    Example:
        Location(type="airport", value="CDG", countryCode="FR")
        Location(type="address", value="5 Avenue Anatole France, Paris, France", countryCode="FR")
    """
    type: Literal["airport", "address"]
    value: str
    countryCode: str | None = None

    def _get_geo_codes(self) -> str:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": self.value,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "amadeus-agent"
        }

        res = requests.get(url, params=params, headers=headers)
        data = res.json()

        if not data:
            return "48.858093,2.294694"  # Default fallback

        return f"{data[0]['lat']},{data[0]['lon']}"

    def to_api(self, locationType: Literal["start", "end"]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.type == "airport":
            data[f"{locationType}LocationCode"] = self.value
        elif self.type == "address":
            data[f"{locationType}AddressLine"] = self.value
            data[f"{locationType}GeoCode"] = self._get_geo_codes()

        if self.countryCode:
            data[f"{locationType}CountryCode"] = self.countryCode

        return data


class TransferSearch(BaseModel):
    """
    Schema for transfer search tool inputs.
    """
    start: Location
    end: Location
    startDateTime: datetime
    passengers: int = 1
    transferType: Literal["PRIVATE", "SHARED"] = "PRIVATE"
    offersCount: int = 5
    pageCount: int = 1


class PassengerContacts(BaseModel):
    """
    Schema for passenger contact details.
    """
    phoneNumber: str = Field(
        ...,
        description="Phone number with country code, e.g. +33123456789"
    )
    email: EmailStr


class Passenger(BaseModel):
    """
    Schema for a passenger.
    """
    firstName: str
    lastName: str
    title: Literal["MR", "MRS", "MS", "MISS"]
    contacts: PassengerContacts


class Payment(BaseModel):
    """
    Schema for payment details.
    """
    methodOfPayment: Literal["INVOICE"] = "INVOICE"


class TransferBooking(BaseModel):
    """
    Schema for transfer booking tool inputs.
    """
    offerId: str
    passengers: list[Passenger]
    payment: Payment


class TransferCancellation(BaseModel):
    """
    Schema for transfer cancellation tool inputs.
    """
    orderId: str
    confirmationNumber: str


class UserData(BaseModel):
    """
    Schema for user data stored in the agent context.
    """
    title: Literal["MR", "MRS", "MS", "MISS"] | None = None
    first_name: str | None = None
    email: EmailStr | None = None
    last_name: str | None = None
    phone: str | None = None


class NewPassengers(BaseModel):
    """
    Schema for new passengers data stored in the agent context.
    """
    passengers: list[Passenger] = []

# -------------------------------
# 2. Tools using Amadeus API
# -------------------------------


@tool(args_schema=NewPassengers)
async def add_passengers(
    passengers: list[Passenger],
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    """
    Add passengers to the user data from the agent context.

    :param passengers: Description
    :type passengers: list[Passenger]
    :param runtime: Description
    :type runtime: ToolRuntime[UserSessionContext]
    :return: Description
    :rtype: str
    """
    logger.info("Adding %d passenger(s) for agent_id=%s",
                len(passengers), runtime.context.agent_id)
    store = runtime.store

    logger.info("Loading existing user data from store")
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )

    if not user_data:
        data = {
            "passengers": {},
            "orders": []
        }
    else:
        data = user_data.value

    existing_passengers = data.get("passengers")

    for p in passengers:
        logger.info("Adding passenger: %s %s %s",
                     p.title, p.firstName, p.lastName)
        pid = None
        while pid is None or pid in existing_passengers:
            pid = f"passenger_{randint(1000, 9999)}"
        existing_passengers[pid] = p.model_dump()

    data["passengers"] = existing_passengers

    logger.info("Saving updated passenger list to store")
    await store.aput(
        (runtime.context.agent_id,),
        "user_data",
        data
    )

    logger.info("Successfully added %d passenger(s). Total now: %d",
                len(passengers), len(existing_passengers))
    return f"Added {len(passengers)} passengers. Total now: {len(existing_passengers)}."


@tool
async def list_passengers(
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    """
    List passengers from the user data in the agent context.

    :param runtime: Description
    :type runtime: ToolRuntime[UserSessionContext]
    :return: Description
    :rtype: str
    """
    logger.info("Listing passengers for agent_id=%s", runtime.context.agent_id)
    store = runtime.store

    logger.info("Loading user data from store")
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )
    if not user_data:
        data = {
            "passengers": {},
            "orders": []
        }
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
        existing_passengers = {}
        logger.info("No user data found, initialized new user data")
    else:
        existing_passengers = user_data.value.get("passengers")

    if existing_passengers is None or len(existing_passengers) == 0:
        logger.info("No passengers found for agent_id=%s",
                    runtime.context.agent_id)
        return "No passengers found."

    logger.info("Found %d passenger(s) for agent_id=%s", len(
        existing_passengers), runtime.context.agent_id)
    res = "Passengers:\n"
    for pid, p in existing_passengers.items():
        res += (
            f"- ID: {pid} (dont'show pid to user), Name: {p['title']} {p['firstName']} {p['lastName']}, "
            f"Email: {p['contacts']['email']}, Phone: {p['contacts']['phoneNumber']}\n"
        )

    return res


@tool
async def remove_passenger(
    passenger_id: str,
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    """
    Remove a passenger from the user data in the agent context.

    :param passenger_id: ID of the passenger to remove
    :type passenger_id: str
    :param runtime: Description
    :type runtime: ToolRuntime[UserSessionContext]
    :return: Description
    :rtype: str
    """
    logger.info("Removing passenger %s for agent_id=%s",
                passenger_id, runtime.context.agent_id)
    store = runtime.store

    logger.info("Loading user data from store")
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )
    if not user_data:
        logger.warning("No user data found for agent_id=%s",
                       runtime.context.agent_id)
        return "No passengers found."

    existing_passengers = user_data.value.get("passengers")
    if existing_passengers is None or passenger_id not in existing_passengers:
        logger.info("Passenger %s not found for agent_id=%s",
                    passenger_id, runtime.context.agent_id)
        return f"Passenger {passenger_id} not found."

    del existing_passengers[passenger_id]
    user_data.value["passengers"] = existing_passengers

    logger.info("Saving updated passenger list to store")
    await store.aput(
        (runtime.context.agent_id,),
        "user_data",
        user_data.value
    )

    logger.info("Successfully removed passenger %s. Total now: %d",
                passenger_id, len(existing_passengers))
    return f"Removed passenger {passenger_id}. Total now: {len(existing_passengers)}."


@tool
async def list_active_orders(
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    """
    List active transfer orders for the user.

    :param runtime: Description
    :type runtime: ToolRuntime[UserSessionContext]
    :return: Description
    :rtype: str
    """

    store = runtime.store
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )

    if not user_data:
        logger.info("No user data found, initializing new user data")
        data = {
            "passengers": {},
            "orders": []
        }
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
        logger.info("No orders found for agent_id=%s",
                    runtime.context.agent_id)
        return "No orders found."

    existing_orders = user_data.value.get("orders", [])
    active_orders = [order for order in existing_orders if order.get(
        "status") == "SUCCESS" and order["offer"]["startDateTime"] >= datetime.now().isoformat()]
    if not active_orders:
        logger.info("No active orders found for agent_id=%s",
                    runtime.context.agent_id)
        return "No active orders found."

    logger.info("Found %d active order(s) for agent_id=%s",
                len(active_orders), runtime.context.agent_id)
    res = "Active Orders:\n"
    for idx, order in enumerate(active_orders, start=1):
        offer = order["offer"]
        res += (
            f"{idx}. Order ID: {order.get('orderId')}, Confirmation Number: {order.get('confirmationNumber')}, "
            f"Pickup: {offer.get('pickupLocation')}, Dropoff: {offer.get('dropoffLocation')}, "
            f"Pickup DateTime: {offer.get('startDateTime')}, Vehicle Type: {offer.get('vehicleType')}, "
            f"Provider: {offer.get('provider')}, Price: {offer.get('price')} {offer.get('currency')}\n"
        )

    return res


@tool(args_schema=TransferSearch)
async def search_transfers(
    start: Location,
    end: Location,
    startDateTime: datetime,
    runtime: ToolRuntime[UserSessionContext],
    passengers: int = 1,
    transferType: Literal["PRIVATE", "SHARED"] = "PRIVATE",
    offersCount: int = 5,
    pageCount: int = 1,
) -> str:
    logger.info(
        "Searching transfers: %s -> %s, datetime=%s, passengers=%d, type=%s",
        start.value, end.value, startDateTime, passengers, transferType
    )
    client = CLIENT

    start_location = start.to_api("start")
    end_location = end.to_api("end")

    req = TransferSearchRequest(
        startDateTime=startDateTime,
        passengers=passengers,
        transferType=transferType,
        **start_location,
        **end_location,
    )
    logger.info("Transfer search request: %s", req.to_api())

    logger.info("Calling Amadeus API for transfer search")
    response = await client.search_transfers(req)
    logger.info("Received response from Amadeus API")

    response.only_invoice()
    response.best_offer(count=offersCount, page=pageCount)
    logger.info("Filtered to %d best offer(s) with invoice payment", len(response.to_dict()))

    store = runtime.store

    logger.info("Loading previous search responses from store")
    transfer_search_responses = await store.aget(
        (runtime.context.thread_id, runtime.context.agent_id),
        "transfer_search_responses"
    )

    if not transfer_search_responses:
        data = response.to_dict()
        logger.info("No previous search responses, creating new entry")
    else:
        data = transfer_search_responses.value
        data.update(response.to_dict())
        logger.info("Merged with existing search responses")

    logger.info(f"Saving search responses to store {data}")
    await store.aput(
        (runtime.context.thread_id, runtime.context.agent_id),
        "transfer_search_responses",
        data
    )

    logger.info(f"Transfer search completed successfully - {response}")
    return str(response)


@tool(args_schema=TransferBooking)
async def book_transfer(
    offerId: str,
    passengers: list[Passenger],
    payment: Payment,
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    logger.info("Booking transfer: offerId=%s, passengers=%d, payment=%s",
                offerId, len(passengers), payment.methodOfPayment)
    client = CLIENT
    store = runtime.store

    logger.info("Preparing booking data with %d passenger(s)",
                 len(passengers))
    booking_data: dict[str, Any] = {
        "data": {
            "passengers": [
                {
                    "firstName": p.firstName,
                    "lastName": p.lastName,
                    "title": p.title,
                    "contacts": {
                        "phoneNumber": p.contacts.phoneNumber,
                        "email": p.contacts.email,
                    },
                } for p in passengers
            ],
            "payment": {
                "methodOfPayment": payment.methodOfPayment
            }
        }
    }

    logger.info("Loading transfer search responses from store")
    transfer_search_responses = await store.aget(
        (runtime.context.thread_id, runtime.context.agent_id),
        "transfer_search_responses"
    )

    if not transfer_search_responses:
        logger.error("No transfer search responses found for booking")
        return "Error: No transfer search responses found. Please search for transfers before booking."

    order_data = transfer_search_responses.value.get(offerId)
    if order_data:
        booking_data["offer"] = order_data
        logger.info("Found offer data for offerId=%s", offerId)
    else:
        logger.warning("No offer data found for offerId=%s", offerId)

    req = TransferBookingRequest(
        offerId=offerId,
        data=booking_data
    )

    logger.info("Calling Amadeus API for transfer booking")
    response = await client.book_transfer(req)

    if not response.is_success:
        logger.error("Booking failed for offerId=%s", offerId)
        res = ""
        for err in response.errors:
            if isinstance(err, dict):
                logger.error("Booking error: [%s] %s: %s", err.get(
                    'code'), err.get('title'), err.get('detail'))
                res += f"[{err.get('code')}] {err.get('title')}: {err.get('detail')}\n"
            else:
                logger.error(
                    "Booking error: [%s] %s: %s", err.code, err.title, err.detail)
                res += f"[{err.code}] {err.title}: {err.detail}\n"
        return res

    data = response.data or {}

    if isinstance(data, dict):
        order_id = data.get("id")
        transfers = data.get("transfers", [])
    else:
        order_id = getattr(data, "id", None)
        transfers = getattr(data, "transfers", []) or []

    confirm_nbr = None
    if transfers:
        first_transfer = transfers[0]
        if isinstance(first_transfer, dict):
            confirm_nbr = first_transfer.get("confirmNbr")
        else:
            confirm_nbr = getattr(first_transfer, "confirmNbr", None)

    booking_data["orderId"] = order_id
    booking_data["confirmationNumber"] = confirm_nbr
    booking_data["status"] = "SUCCESS"
    logger.info(
        "Booking successful: orderId=%s, confirmationNumber=%s", order_id, confirm_nbr)

    logger.info("Loading user data to save order")
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )

    existing_orders = user_data.value.get("orders")
    if existing_orders is None:
        existing_orders = []
        logger.info("No existing orders, creating new list")

    existing_orders.append(booking_data)
    user_data.value["orders"] = existing_orders
    logger.info("Saving order to user data, total orders: %d",
                 len(existing_orders))

    await store.aput(
        (runtime.context.agent_id,),
        "user_data",
        user_data.value
    )
    logger.info("Order saved successfully")

    subject, body = render_transfer_confirmation_email(
        customer_name=", ".join(
            [
                f"{p['title'].title()} {p['firstName']} {p['lastName']}"
                for p in booking_data.get("data", {}).get("passengers", [])
            ]
        ) or "N/A",
        order_id=order_id or "N/A",
        confirmation_number=confirm_nbr or "N/A",
        pickup_datetime=booking_data.get(
            "offer", {}).get("startDateTime", "N/A"),
        pickup_location=booking_data.get(
            "offer", {}).get("startLocation", "N/A"),
        dropoff_location=booking_data.get(
            "offer", {}).get("endLocation", "N/A"),
        provider=booking_data.get("offer", {}).get("provider", "N/A"),
        vehicle_type=booking_data.get("offer", {}).get("vehicle", "N/A"),
        price=booking_data.get("offer", {}).get("price", "N/A"),
        cancellation_policy=booking_data.get(
            "offer", {}).get("cancellation", "N/A"),
    )

    logger.info("Sending confirmation email to %d recipient(s)",
                 len(passengers))
    to_set = set([passenger.contacts.email for passenger in passengers])
    send_email(
        to=list(to_set),
        from_address=f"{os.environ.get('EMAIL_SENDER_NAME', 'Amadeus Transfers Agent')} <confirmation.transfers@{os.environ.get('EMAIL_DOMAIN')}>",
        subject=subject,
        html_content=body
    )
    logger.info("Confirmation email sent for orderId=%s", order_id)

    res = f"Booking successful! Order ID: {order_id}, Confirmation Number: {confirm_nbr}"
    return res


@tool(args_schema=TransferCancellation)
async def cancel_transfer(
    orderId: str,
    confirmationNumber: str,
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    logger.info("Cancelling transfer: orderId=%s, confirmationNumber=%s",
                orderId, confirmationNumber)
    client = CLIENT
    store = runtime.store

    logger.info("Calling Amadeus API for transfer cancellation")
    response = await client.cancel_transfer(
        order_id=orderId,
        confirm_nbr=confirmationNumber
    )

    if "errors" in response:
        logger.error("Cancellation failed for orderId=%s", orderId)
        res = ""
        for err in response["errors"]:
            logger.error("Cancellation error: [%s] %s: %s", err.get(
                'code'), err.get('title'), err.get('detail'))
            res += f"[{err.get('code')}] {err.get('title')}: {err.get('detail')}\n"
        return res

    logger.info("Updating order status in user data")
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )
    if not user_data:
        logger.warning("No user data found for agent_id=%s",
                       runtime.context.agent_id)
        data = {
            "passengers": {},
            "orders": []
        }
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
    else:
        # Update the order status in existing user data
        orders = user_data.value.get("orders", [])
        for order in orders:
            if order.get("orderId") == orderId and order.get("confirmationNumber") == confirmationNumber:
                order["status"] = "CANCELLED"
                break
        user_data.value["orders"] = orders
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            user_data.value
        )

        subject, body = render_transfer_cancellation_email(
            pickup_datetime=order.get("offer", {}).get("startDateTime", "N/A"),
            order_id=orderId,
            confirmation_number=confirmationNumber,
            pickup_location=order.get("offer", {}).get("startLocation", "N/A"),
            dropoff_location=order.get("offer", {}).get("endLocation", "N/A"),
            provider=order.get("offer", {}).get("provider", "N/A"),
            customer_name=", ".join(
                [
                    f"{p['title'].title()} {p['firstName']} {p['lastName']}"
                    for p in order.get("data", {}).get("passengers", [])
                ]
            ) or "N/A",
            vehicle_type=order.get("offer", {}).get("vehicle", "N/A"),
        )
        logger.info("Sending cancellation email")
        send_email(
            to=list(set([p['contacts']['email']
                for p in order.get("data", {}).get("passengers", [])])),
            from_address=f"{os.environ.get('EMAIL_SENDER_NAME', 'Amadeus Transfers Agent')} <cancellation.transfers@{os.environ.get("EMAIL_DOMAIN")}>",
            subject=subject,
            html_content=body
        )
        logger.info("Cancellation email sent for orderId=%s", orderId)

    logger.info("Transfer cancellation successful: orderId=%s", orderId)
    return "Transfer cancellation successful!"


class AgentAI:

    TOOLS = [
        search_transfers,
        book_transfer,
        cancel_transfer,
        add_passengers,
        list_passengers,
        list_active_orders,
        remove_passenger
    ]

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None
        self._initialized = False
        self._agent: Any = None

    async def __aenter__(self) -> "AgentAI":
        await self._setup_agent()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._checkpointer = None
        self._store = None
        self._initialized = False
        self._agent = None

    async def _ensure_pool(self) -> None:
        if self._pool is None:
            self._pool = AsyncConnectionPool(
                self._get_database_uri(),
                kwargs={"row_factory": dict_row},
                open=False
            )
            await self._pool.open()
            self._checkpointer = AsyncPostgresSaver(self._pool)
            self._store = AsyncPostgresStore(self._pool)

    async def _run_setup_ddl(self) -> None:
        """Run setup DDL with autocommit to allow CREATE INDEX CONCURRENTLY."""
        async with await AsyncConnection.connect(
            self._get_database_uri(),
            autocommit=True
        ) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            store = AsyncPostgresStore(conn)
            await checkpointer.setup()
            await store.setup()

    def _get_database_uri(self) -> str:
        DB_USER = os.environ.get("POSTGRES_USER")
        DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
        DB_HOST = os.environ.get("POSTGRES_HOST")
        DB_PORT = os.environ.get("POSTGRES_PORT")
        DB_NAME = os.environ.get("POSTGRES_DB")

        DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return DB_URI

    async def _setup_agent(self) -> None:
        if not self._initialized:
            await self._run_setup_ddl()
            await self._ensure_pool()
            self._initialized = True

        PROMPT_PATH = Path(__file__).parent / "prompt.md"

        self._agent = create_agent(
            model=os.environ.get("AGENT_MODEL", "gpt-4o"),
            tools=self.TOOLS,
            checkpointer=self._checkpointer,
            store=self._store,
            context_schema=UserSessionContext,
            system_prompt=PROMPT_PATH.read_text(),
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "book_transfer": {"allowed_decisions": ["approve", "reject"]},  # Requires approval before booking
                        # Other tools don't need approval
                        "cancel_transfer": False,
                        "search_transfers": False,
                        "add_passengers": False,
                        "list_passengers": False,
                        "list_active_orders": False,
                        "remove_passenger": False,
                    },
                    description_prefix="Payment confirmation required",
                ),
            ],
        )

    async def ask_agent(
        self,
        thread_id: str,
        agent_id: str,
        question: str
    ) -> dict[str, Any]:
        """
        Process agent request and handle interrupts for payment.

        Returns:
            - If interrupted: {"status": "interrupted", "payment_required": {...}}
            - If completed: {"status": "completed", "message": "..."}
        """
        if not self._agent:
            await self._setup_agent()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = await self._agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config,
            context=UserSessionContext(
                thread_id=thread_id,
                agent_id=agent_id
            )
        )

        # Check for interrupt (payment needed)
        if "__interrupt__" in result:

            print(result["__interrupt__"])

            interrupt_data = result["__interrupt__"][0].value
            action_request = interrupt_data["action_requests"][0]

            # Extract booking details - use "args" not "arguments"
            offer_id = action_request["args"]["offerId"]
            passengers = action_request["args"]["passengers"]

            # Fetch offer details from store to get price
            store = self._store
            transfer_search_responses = await store.aget(
                (thread_id, agent_id),
                "transfer_search_responses"
            )

            payment_info = None
            if transfer_search_responses:
                offer_data = transfer_search_responses.value.get(offer_id)
                if offer_data:
                # Build payment structure
                    payment_info = {
                        "amount": offer_data.get("price"),
                        "amount_usdc": offer_data.get("price"),
                        "currency": offer_data.get("currency"),
                        "offer_id": offer_id,
                        "passenger_count": len(passengers),
                        "passengers": [
                        {
                            "name": f"{p['title']} {p['firstName']} {p['lastName']}",
                            "email": p['contacts']['email'],
                            "phone": p['contacts']['phoneNumber']
                        }
                        for p in passengers
                        ],
                        "pickup_location": offer_data.get("startLocation"),
                        "dropoff_location": offer_data.get("endLocation"),
                        "pickup_datetime": offer_data.get("startDateTime"),
                        "vehicle_type": offer_data.get("vehicle"),
                        "provider": offer_data.get("provider"),
                    }

            return {
                "status": "interrupted",
                "payment_required": payment_info
            }

        return {
            "status": "completed",
            "message": str(result["messages"][-1].content)
        }

    async def confirm_payment(
        self,
        thread_id: str,
        agent_id: str,
        approved: bool,
        rejection_reason: str | None = None
    ) -> dict[str, Any]:
        """Resume agent execution after payment confirmation."""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        if approved:
            # Payment went through - approve the booking
            result = await self._agent.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config,
                context=UserSessionContext(
                    thread_id=thread_id,
                    agent_id=agent_id
                )
            )
        else:
            # Payment failed - reject the booking
            result = await self._agent.ainvoke(
                Command(
                    resume={
                        "decisions": [{
                            "type": "reject",
                            "message": rejection_reason or "Payment processing failed"
                        }]
                    }
                ),
                config,
                context=UserSessionContext(
                    thread_id=thread_id,
                    agent_id=agent_id
                )
            )

        return {
            "status": "completed",
            "message": str(result["messages"][-1].content)
        }


AI = AgentAI()
