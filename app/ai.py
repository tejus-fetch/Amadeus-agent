import os
from datetime import datetime
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver as PostgresSaver
from pydantic import BaseModel, EmailStr, Field

from app.api import CLIENT, TransferBookingRequest, TransferSearchRequest

# -------------------------------
# 1. Custom Agent State
# -------------------------------


class CustomAgentState(AgentState):
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
        return "48.858093,2.294694"

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

# -------------------------------
# 2. Tools using Amadeus API
# -------------------------------


@tool(args_schema=TransferSearch)
async def search_transfers(
    start: Location,
    end: Location,
    startDateTime: datetime,
    passengers: int = 1,
    transferType: Literal["PRIVATE", "SHARED"] = "PRIVATE",
    offersCount: int = 5,
    pageCount: int = 1,
) -> str:
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
    print(req.to_api())

    response = await client.search_transfers(req)
    response.only_invoice()
    response.best_offer(count=offersCount, page=pageCount)
    return str(response)


@tool(args_schema=TransferBooking)
async def book_transfer(
    offerId: str,
    passengers: list[Passenger],
    payment: Payment,
) -> str:
    client = CLIENT

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

    req = TransferBookingRequest(
        offerId=offerId,
        data=booking_data
    )

    print(req)

    response = await client.book_transfer(req)

    print(response)

    if not response.is_success:
        res = ""
        for err in response.errors:
            if isinstance(err, dict):
                res += f"[{err.get('code')}] {err.get('title')}: {err.get('detail')}\n"
            else:
                res += f"[{err.code}] {err.title}: {err.detail}\n"
        return res

    data = response.data or {}

    print(data)

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

    res = f"Booking successful! Order ID: {order_id}, Confirmation Number: {confirm_nbr}"
    return res


@tool(args_schema=TransferCancellation)
async def cancel_transfer(
    orderId: str,
    confirmationNumber: str,
) -> str:
    client = CLIENT

    response = await client.cancel_transfer(
        order_id=orderId,
        confirm_nbr=confirmationNumber
    )

    if "errors" in response:
        res = ""
        for err in response["errors"]:
            res += f"[{err.get('code')}] {err.get('title')}: {err.get('detail')}\n"
        return res

    return "Transfer cancellation successful!"


class AgentAI:
    def __init__(self, tools: list[Any]) -> None:
        self._tools = tools

    def _get_database_uri(self) -> str:
        DB_USER = os.environ.get("POSTGRES_USER")
        DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
        DB_HOST = os.environ.get("POSTGRES_HOST")
        DB_PORT = os.environ.get("POSTGRES_PORT")
        DB_NAME = os.environ.get("POSTGRES_DB")

        DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return DB_URI

    async def ask_agent(self, thread_id: str, agent_id: str, question: str) -> str:

        db_uri = self._get_database_uri()
        async with PostgresSaver.from_conn_string(db_uri) as checkpointer:
            await checkpointer.setup()

            with open("app/prompt.md") as f:
                SYSTEM_PROMPT = f.read()

            agent = create_agent(
                model="gpt-4o-mini",
                tools=self._tools,
                state_schema=CustomAgentState,
                checkpointer=checkpointer,
                system_prompt=SYSTEM_PROMPT
            )

            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            result = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ],
                    "agent_id": agent_id,
                },
                config,
            )

            return result["messages"][-1].content


AI = AgentAI([search_transfers, book_transfer, cancel_transfer])