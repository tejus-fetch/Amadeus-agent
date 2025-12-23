import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Any, Literal

import httpx
import requests
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


def _fx_to_usdc(amount: float, currency: str) -> float | None:
    """Convert amount in given currency to USDC using FreeCurrencyAPI; returns string like '12.34' or None on failure."""
    try:
        amt = float(amount)
    except Exception:
        return None
    cur = (currency or "").upper()
    if not cur:
        return None
    if cur in {"USD", "USDC"}:
        return float(f"{amt:.2f}")
    try:
        key = os.getenv("FREECURRENCYAPI_KEY", "")
        if not key:
            return None
        with httpx.Client(timeout=10.0) as client:
            r = client.get("https://api.freecurrencyapi.com/v1/latest",
                           params={"apikey": key, "currencies": cur})
        if r.status_code >= 400:
            return None
        data = r.json() or {}
        rates = data.get("data") or {}
        usd_to_cur = float(rates.get(cur)) if cur in rates else None
        if not usd_to_cur or usd_to_cur <= 0:
            return None
        usd_per_cur = 1.0 / usd_to_cur
        return float(f"{(amt * usd_per_cur):.2f}")
    except Exception:
        return None


def _fet_usdc_price() -> float | None:
    """Fetch FET/USDC price from Binance; returns USDC per 1 FET."""
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                "https://api.binance.com/api/v3/ticker/price", params={"symbol": "FETUSDC"})
        if r.status_code >= 400:
            return None
        data = r.json() or {}
        price = float(data.get("price"))
        return price if price > 0 else None
    except Exception:
        return None


def _usdc_to_fet(amount_usdc: float) -> float | None:
    """Convert a USDC amount to FET using Binance price. Returns a string with 6 decimals, or None."""
    try:
        amt = float(amount_usdc)
        px = _fet_usdc_price()
        if not px or px <= 0:
            return None
        fet = amt / px
        return float(f"{fet:.6f}")
    except Exception:
        return None


def _fet_to_usdc(amount_fet: float) -> float | None:
    """Convert a FET amount to USDC using Binance price. Returns a string with 2 decimals, or None."""
    try:
        amt = float(amount_fet)
        px = _fet_usdc_price()
        if not px or px <= 0:
            return None
        usdc = amt * px
        return float(f"{usdc:.2f}")
    except Exception:
        return None

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
        # Try Google Maps API first
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if api_key:
            try:
                url = "https://maps.googleapis.com/maps/api/geocode/json"
                params: dict[str, str] = {
                    "address": self.value,
                    "key": api_key,
                }

                res = requests.get(url, params=params, timeout=10)
                data = res.json()

                if data.get("status") == "OK" and data.get("results"):
                    location = data["results"][0]["geometry"]["location"]
                    return f"{location['lat']},{location['lng']}"
            except Exception:
                pass  # Fall through to OpenStreetMap

        # Fallback to OpenStreetMap Nominatim
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params: dict[str, str] = {
                "q": self.value,
                "format": "json",
                "limit": "1"
            }
            headers = {
                "User-Agent": "amadeus-agent"
            }

            res = requests.get(url, params=params, headers=headers, timeout=10)
            data = res.json()

            if data:
                return f"{data[0]['lat']},{data[0]['lon']}"
        except Exception:
            pass

        return "48.858093,2.294694"  # Default fallback

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
            "orders": [],
            "balance": 0.0,
            "payments": {}
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
            "orders": [],
            "balance": 0.0,
            "payments": {}
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
            "orders": [],
            "balance": 0.0,
            "payments": {}
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
    try:
        response = await client.search_transfers(req)
        if not response.is_success:
            logger.error("Search failed with errors: %s", response.errors)
            res = ""
            for err in response.errors:
                if isinstance(err, dict):
                    logger.error("Search error: [%s] %s: %s", err.get(
                        'code'), err.get('title'), err.get('detail'))
                    res += f"[{err.get('code')}] {err.get('title')}: {err.get('detail')}\n"
                else:
                    logger.error(
                        "Search error: [%s] %s: %s", err.code, err.title, err.detail)
                    res += f"[{err.code}] {err.title}: {err.detail}\n"
            return res + "Please modify your search criteria and try again. (Don't show internal error details to user)"
    except Exception as e:
        logger.error("Transfer search failed: %s", e)
        return f"Error: Transfer search failed due to {e}: Don't tell user the internal error details. Please try again later."
    logger.info("Received response from Amadeus API")

    response.only_invoice()
    response.best_offer(count=offersCount, page=pageCount)
    logger.info("Filtered to %d best offer(s) with invoice payment",
                len(response.to_dict()))

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
    return response.to_str()


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

    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )
    if not user_data:
        logger.warning("No user data found for agent_id=%s",
                       runtime.context.agent_id)
        data = {
            "passengers": {},
            "orders": [],
            "balance": 0.0,
            "payments": {}
        }
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
        return "Error: No user data found. Please try again."
    else:
        data = user_data.value

    payment_ctx = data.get("payments", {}).get(f"{runtime.context.agent_id}:{runtime.context.thread_id}:{offerId}", None)
    if not payment_ctx:
        logger.error("No payment context found for booking")
        return "Error: No payment context found. Please initiate payment before booking."

    logger.info("Loading transfer search responses from store")
    transfer_search_responses = await store.aget(
        (runtime.context.thread_id, runtime.context.agent_id),
        "transfer_search_responses"
    )

    if not transfer_search_responses:
        logger.error("No transfer search responses found for booking")

        data["balance"] += payment_ctx.get("total_fet", 0.0)

        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
        await store.aput(
            (runtime.context.thread_id, runtime.context.agent_id),
            "transfer_search_responses",
            {}
        )

        return f"Error: No transfer search responses found. Please search for transfers before booking. Your payment of {payment_ctx.get('total_fet', 0.0)} FET has been refunded to your balance."

    order_data = transfer_search_responses.value.get(offerId)
    if order_data:
        booking_data["offer"] = order_data
        logger.info("Found offer data for offerId=%s", offerId)
    else:
        logger.warning("No offer data found for offerId=%s", offerId)

        data["balance"] += payment_ctx.get("total_fet", 0.0)
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )

        return f"Error: No order data found for offerId={offerId}. Please search for transfers before booking. Your payment of {payment_ctx.get('total_fet', 0.0)} FET has been refunded to your balance."

    req = TransferBookingRequest(
        offerId=offerId,
        data=booking_data
    )

    logger.info("Calling Amadeus API for transfer booking")
    try:
        response = await client.book_transfer(req)
    except Exception as e:
        logger.error("Transfer booking failed: %s", e)

        data["balance"] += payment_ctx.get("total_fet", 0.0)

        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )

        return f"Error: Transfer booking failed. Please try again later. Your payment of {payment_ctx.get('total_fet', 0.0)} FET has been refunded to your balance."

    if not response.is_success:
        logger.error("Booking failed for offerId=%s", offerId)
        error = ""
        for err in response.errors:
            if isinstance(err, dict):
                logger.error("Booking error: [%s] %s: %s", err.get(
                    'code'), err.get('title'), err.get('detail'))
                error += f"[{err.get('code')}] {err.get('title')}: {err.get('detail')}\n"
            else:
                logger.error(
                    "Booking error: [%s] %s: %s", err.code, err.title, err.detail)
                error += f"[{err.code}] {err.title}: {err.detail}\n"

        logger.error("Booking failed errors: %s", error)

        data["balance"] += payment_ctx.get("total_fet", 0.0)
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
        return f"Error: Transfer booking failed:\n{error}, Don't tell user the internal error details. Please try again later. Your payment of {payment_ctx.get('total_fet', 0.0)} FET has been refunded to your balance."

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
    booking_data["bookedAt"] = datetime.now().isoformat()
    booking_data["totalPaidFET"] = payment_ctx.get("total_fet", 0.0)
    logger.info(
        "Booking successful: orderId=%s, confirmationNumber=%s", order_id, confirm_nbr)

    logger.info("Loading user data to save order")

    existing_orders = data.get("orders")
    if existing_orders is None:
        existing_orders = []
        logger.info("No existing orders, creating new list")

    existing_orders.append(booking_data)
    data["orders"] = existing_orders
    logger.info("Saving order to user data, total orders: %d",
                len(existing_orders))

    await store.aput(
        (runtime.context.agent_id,),
        "user_data",
        data
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
    try:
        response = await client.cancel_transfer(
            order_id=orderId,
            confirm_nbr=confirmationNumber
        )
    except Exception as e:
        logger.error("Transfer cancellation failed: %s", e)
        return f"Error: Transfer cancellation failed due to {e}: Don't tell user the internal error details. Please try again later."

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
            "orders": [],
            "balance": 0.0,
            "payments": {}
        }
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
        logger.info("No user data found, initialized new user data")
        return "No orders found to cancel."
    else:
        # Update the order status in existing user data
        orders = user_data.value.get("orders", [])
        refundable_amount_fet = 0.0
        for order in orders:
            if order.get("orderId") == orderId and order.get("confirmationNumber") == confirmationNumber:
                order["status"] = "CANCELLED"
                order["cancelledAt"] = datetime.now().isoformat()
                refundable_amount_fet = order.get("totalPaidFET", 0.0)

                # Refund the amount to user's balance
                available_balance = user_data.value.get("balance", 0.0)
                user_data.value["balance"] = available_balance + refundable_amount_fet
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
            refund_status=f"Refunded {refundable_amount_fet} FET to your account balance." if refundable_amount_fet > 0 else "No refund applicable.",
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


@tool
async def get_current_datetime() -> str:
    """Get the current date and time in ISO format."""
    logger.info("Fetching current date and time")
    currentDatetime = datetime.now().isoformat()
    logger.info("Current date and time: %s", currentDatetime)
    return currentDatetime

@tool
async def get_user_balance(
    runtime: ToolRuntime[UserSessionContext],
) -> str:
    """
    Get the user's current balance from the user data in the agent context.

    :param runtime: Description
    :type runtime: ToolRuntime[UserSessionContext]
    :return: Description
    :rtype: str
    """
    logger.info("Fetching user balance for agent_id=%s", runtime.context.agent_id)
    store = runtime.store

    logger.info("Loading user data from store")
    user_data = await store.aget(
        (runtime.context.agent_id,),
        "user_data"
    )
    if not user_data:
        logger.info("No user data found for agent_id=%s", runtime.context.agent_id)

        data = {
            "passengers": {},
            "orders": [],
            "balance": 0.0,
            "payments": {}
        }
        await store.aput(
            (runtime.context.agent_id,),
            "user_data",
            data
        )
    else:
        data = user_data.value

    balance = data.get("balance", 0.0)
    logger.info("User balance is %f FET", balance)
    return f"Your current balance is {balance:.2f} FET."


class AgentAI:

    TOOLS = [
        search_transfers,
        book_transfer,
        cancel_transfer,
        add_passengers,
        list_passengers,
        list_active_orders,
        remove_passenger,
        get_current_datetime,
        get_user_balance,
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
                        # Requires approval before booking
                        "book_transfer": {"allowed_decisions": ["approve", "reject"]},
                        # Other tools don't need approval
                        "cancel_transfer": False,
                        "search_transfers": False,
                        "add_passengers": False,
                        "list_passengers": False,
                        "list_active_orders": False,
                        "remove_passenger": False,
                        "get_current_datetime": False,
                        "get_user_balance": False,
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

            if not transfer_search_responses:
                logger.error(
                    "No transfer search responses found in store during interrupt handling")
                await store.aput(
                    (thread_id, agent_id),
                    "transfer_search_responses",
                    {}
                )
                return {
                    "status": "completed",
                    "message": "No transfer search responses found. Please search for transfers before booking."
                }

            user_data = await store.aget(
                (agent_id,),
                "user_data"
            )
            if not user_data:
                logger.error(
                    "No user data found in store during interrupt handling")
                await store.aput(
                    (agent_id,),
                    "user_data",
                    {
                        "passengers": {},
                        "orders": [],
                        "balance": 0.0,
                        "payments": {}
                    }
                )
                return {
                    "status": "completed",
                    "message": "No user data found. Please add passengers before booking."
                }

            data = user_data.value
            balance = data.get("balance", 0.0)

            payment_info = None
            if transfer_search_responses:
                offer_data = transfer_search_responses.value.get(offer_id)
                if offer_data:
                    # Build payment structure
                    amount, currency = offer_data.get("price").split(" ")

                    amount = float(amount) * 0.001

                    amount_usdc = _fx_to_usdc(amount, currency)

                    if not amount_usdc:
                        logger.error(
                            "Unsupported currency %s for offerId=%s during interrupt handling", currency, offer_id)
                        return {
                            "status": "completed",
                            "message": f"Unsupported currency {currency} for offerId={offer_id}. Please contact support."
                        }

                    amount_fet = _usdc_to_fet(amount_usdc)

                    if not amount_fet:
                        logger.error(
                            "Failed to convert amount to FET for offerId=%s during interrupt handling", offer_id)
                        return {
                            "status": "completed",
                            "message": f"Failed to process payment for offerId={offer_id}. Please contact support."
                        }

                    if balance > 0.0:
                        if amount_fet > balance:
                            needed_amount_fet = float(amount_fet) - balance
                            needed_amount_usdc = _fet_to_usdc(needed_amount_fet)
                            data["balance"] = 0.0
                            payment_info = {
                                "total_fet": amount_fet,
                                "amount": amount,
                                "currency": currency,
                                "amount_usdc": needed_amount_usdc,
                                "amount_fet": needed_amount_fet,
                                "paid_fet": balance,
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
                            data["payments"][f"{agent_id}:{thread_id}:{offer_id}"] = payment_info
                            await store.aput(
                                (agent_id,),
                                "user_data",
                                data
                            )

                            return {
                                "status": "interrupted",
                                "payment_required": payment_info
                            }

                        else:
                            balance -= float(amount_fet)
                            data["balance"] = balance
                            payment_info = {
                                "total_fet": amount_fet,
                                "amount": 0.0,
                                "currency": currency,
                                "amount_usdc": 0.0,
                                "amount_fet": 0.0,
                                "paid_fet": amount_fet,
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
                            data["payments"][f"{agent_id}:{thread_id}:{offer_id}"] = payment_info

                            await store.aput(
                                (agent_id,),
                                "user_data",
                                data
                            )

                            result = await self.confirm_payment(
                                thread_id=thread_id,
                                agent_id=agent_id,
                                approved=True
                            )

                            return {
                                "status": "completed",
                                "message": str(result["message"])
                            }

                    else:

                        payment_info = {
                            "total_fet": amount_fet,
                            "amount": float(amount),
                            "currency": currency,
                            "amount_usdc": amount_usdc,
                            "amount_fet": amount_fet,
                            "paid_fet": 0.0,
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
                        data["payments"][f"{agent_id}:{thread_id}:{offer_id}"] = payment_info
                        await store.aput(
                            (agent_id,),
                            "user_data",
                            data
                        )

                        return {
                            "status": "interrupted",
                            "payment_required": payment_info
                        }
                else:
                    logger.error(
                        "No offer data found for offerId=%s during interrupt handling", offer_id)
                    return {
                        "status": "completed",
                        "message": f"No offer data found for offerId={offer_id}. Please search for transfers before booking."
                    }
            else:
                logger.error(
                    "No offer data found for offerId=%s during interrupt handling", offer_id)
                return {
                    "status": "completed",
                    "message": f"No offer data found for offerId={offer_id}. Please search for transfers before booking."
                }
        else:
            return {
                "status": "completed",
                "message": str(result["messages"][-1].content)
            }

    async def confirm_payment(
        self,
        thread_id: str,
        agent_id: str,
        approved: bool,
        payment_ctx: dict[str, Any] | None = None,
        rejection_reason: str | None = None
    ) -> dict[str, Any]:
        """Resume agent execution after payment confirmation."""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        store = self._store
        pre_paid_amount = payment_ctx.get("payment_info", {}).get(
            "paid_fet", 0.0) if payment_ctx else 0.0

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

            if pre_paid_amount and pre_paid_amount > 0.0:
                # Refund pre-paid amount to user balance
                user_data = await store.aget(
                    (agent_id,),
                    "user_data"
                )
                if not user_data:
                    data = {
                        "passengers": {},
                        "orders": [],
                        "balance": pre_paid_amount,
                        "payments": {}
                    }
                else:
                    data = user_data.value
                    balance = data.get("balance", 0.0)
                    balance += pre_paid_amount
                    data["balance"] = balance
                await store.aput(
                    (agent_id,),
                    "user_data",
                    data
                )

                result = await self._agent.ainvoke(
                    Command(
                        resume={
                            "decisions": [{
                                "type": "reject",
                                "message": f"Payment verification failed. Refunded amount: {pre_paid_amount} to user balance. Reason: {rejection_reason or 'N/A'}"
                            }]
                        }
                    ),
                    config,
                    context=UserSessionContext(
                        thread_id=thread_id,
                        agent_id=agent_id
                    )
                )
            else:
                result = await self._agent.ainvoke(
                    Command(
                        resume={
                            "decisions": [{
                                "type": "reject",
                                "message": f"Payment verification failed. Reason: {rejection_reason or 'N/A'}"
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
