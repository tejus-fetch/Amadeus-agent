# Async, fully-typed, cached Amadeus Transfers SDK
# Python 3.11+

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

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

# ============================================================
# Base Model
# ============================================================


class LooseBaseModel(BaseModel):
    """Base model for all API responses (intentionally loose)."""

    model_config = {
        "extra": "allow",
        "populate_by_name": True,
    }

# ============================================================
# Utility Models
# ============================================================


class GeoCode(BaseModel):
    """Geographic coordinate."""

    latitude: float
    longitude: float

    def to_api(self) -> str:
        """Convert to Amadeus API format 'latitude,longitude'."""
        return f"{self.latitude},{self.longitude}"


# ============================================================
# Transfer Type Definitions
# ============================================================

TransferType = Literal[
    "PRIVATE", "SHARED", "TAXI", "HOURLY", "AIRPORT_EXPRESS", "AIRPORT_BUS"
]

VehicleCode = Literal[
    "CAR", "SED", "WGN", "ELC", "VAN", "SUV", "LMS", "MBR", "TRN", "BUS"
]

VehicleCategory = Literal["ST", "BU", "FC"]

MethodOfPayment = Literal[
    "CREDIT_CARD", "TRAVEL_ACCOUNT", "PAYMENT_SERVICE_PROVIDER", "INVOICE"
]


# ============================================================
# Search Models
# ============================================================

class PassengerCharacteristics(LooseBaseModel):
    """Passenger demographic information."""

    passengerTypeCode: str  # ADT, CHD, etc.
    age: int | None = None


class StopOverRequest(LooseBaseModel):
    """Stop over location request."""

    duration: str | None = None  # ISO8601 format e.g. PT2H30M
    sequenceNumber: int | None = None
    locationCode: str | None = None
    addressLine: str | None = None
    countryCode: str | None = None
    cityName: str | None = None
    zipCode: str | None = None
    geoCode: str | None = None  # "lat,lon" format
    name: str | None = None
    stateCode: str | None = None
    googlePlaceId: str | None = None
    uicCode: str | None = None


class TravelSegmentLocation(LooseBaseModel):
    """Flight or train departure/arrival location."""

    uicCode: str | None = None
    iataCode: str | None = None
    localDateTime: str | None = None  # ISO8601 format


class TravelSegment(LooseBaseModel):
    """Connected travel segment (flight or train)."""

    transportationType: Literal["FLIGHT", "TRAIN"] | None = None
    transportationNumber: str | None = None
    departure: TravelSegmentLocation | None = None
    arrival: TravelSegmentLocation | None = None


class TransferSearchRequest(LooseBaseModel):
    """
    Typed request model for Transfer Search.

    Based on Amadeus Transfer Search API v1.11.
    """

    # Required fields
    startDateTime: datetime
    startLocationCode: str | None = None

    # Optional passenger info
    passengers: int = Field(default=1, ge=1)
    language: str = "EN"
    currency: str | None = None

    # Start location options
    startUicCode: str | None = None
    startAddressLine: str | None = None
    startCityName: str | None = None
    startZipCode: str | None = None
    startCountryCode: str | None = None
    startStateCode: str | None = None
    startGeoCode: GeoCode | str | None = None
    startName: str | None = None
    startGooglePlaceId: str | None = None

    # End location options
    endLocationCode: str | None = None
    endUicCode: str | None = None
    endAddressLine: str | None = None
    endCityName: str | None = None
    endZipCode: str | None = None
    endCountryCode: str | None = None
    endStateCode: str | None = None
    endGeoCode: GeoCode | str | None = None
    endName: str | None = None
    endGooglePlaceId: str | None = None

    # Transfer options
    transferType: TransferType | None = None
    duration: str | None = None  # ISO8601 format for HOURLY transfers
    vehicleCategory: VehicleCategory | None = None
    vehicleCode: VehicleCode | None = None
    providerCodes: str | None = None
    baggages: int | None = None

    # Extra options
    discountNumbers: str | None = None
    extraServiceCodes: str | None = None
    equipmentCodes: str | None = None
    reference: str | None = None

    # Stop overs and connected segments
    stopOvers: list[StopOverRequest] | None = None
    startConnectedSegment: TravelSegment | None = None
    endConnectedSegment: TravelSegment | None = None

    # Passenger characteristics (note: API has typo "passenegerCharacteristics")
    passengerCharacteristics: list[PassengerCharacteristics] | None = None

    def to_api(self) -> dict[str, Any]:
        """Convert request to Amadeus API payload."""
        data: dict[str, Any] = {}

        # Format datetime
        data["startDateTime"] = self.startDateTime.strftime(
            "%Y-%m-%dT%H:%M:%S")

        # Add all non-None fields
        for field_name, field_value in self.model_dump(exclude_none=True).items():
            if field_name == "startDateTime":
                continue  # Already handled
            elif field_name == "startGeoCode" and field_value:
                if isinstance(self.startGeoCode, GeoCode):
                    data["startGeoCode"] = self.startGeoCode.to_api()
                else:
                    data["startGeoCode"] = field_value
            elif field_name == "endGeoCode" and field_value:
                if isinstance(self.endGeoCode, GeoCode):
                    data["endGeoCode"] = self.endGeoCode.to_api()
                else:
                    data["endGeoCode"] = field_value
            elif field_name == "stopOvers" and field_value:
                data["stopOvers"] = [
                    s.model_dump(exclude_none=True) for s in self.stopOvers or []
                ]
            elif field_name == "startConnectedSegment" and field_value:
                data["startConnectedSegment"] = self.startConnectedSegment.model_dump(
                    exclude_none=True) if self.startConnectedSegment else None
            elif field_name == "endConnectedSegment" and field_value:
                data["endConnectedSegment"] = self.endConnectedSegment.model_dump(
                    exclude_none=True) if self.endConnectedSegment else None
            elif field_name == "passengerCharacteristics" and field_value:
                # Note: API uses "passenegerCharacteristics" (typo in spec)
                data["passenegerCharacteristics"] = [
                    p.model_dump(exclude_none=True)
                    for p in self.passengerCharacteristics or []
                ]
            else:
                data[field_name] = field_value

        return data


# ============================================================
# Search Response Models (loose to handle API variations)
# ============================================================

class PointsAndCash(LooseBaseModel):
    monetaryAmount: str | None = None


class Quotation(LooseBaseModel):
    monetaryAmount: str | None = None
    currencyCode: str | None = None
    isEstimated: bool | None = None
    base: PointsAndCash | None = None
    discount: PointsAndCash | None = None
    totalTaxes: PointsAndCash | None = None
    totalFees: PointsAndCash | None = None


class Seat(LooseBaseModel):
    count: int | None = None
    row: str | None = None
    size: str | None = None


class Baggage(LooseBaseModel):
    count: int | None = None
    size: str | None = None  # S, M, L


class Vehicle(LooseBaseModel):
    code: str | None = None
    category: str | None = None
    description: str | None = None
    seats: list[Seat] | None = None
    baggages: list[Baggage] | None = None
    imageURL: str | None = None


class ServiceProvider(LooseBaseModel):
    code: str | None = None
    name: str | None = None
    logoUrl: str | None = None
    termsUrl: str | None = None
    isPreferred: bool | None = None
    settings: list[str] | None = None


class CancellationRule(LooseBaseModel):
    ruleDescription: str | None = None
    feeType: str | None = None
    feeValue: str | None = None
    currencyCode: str | None = None
    metricType: str | None = None
    metricMin: str | None = None
    metricMax: str | None = None


class Address(LooseBaseModel):
    line: str | None = None
    zip: str | None = None
    countryCode: str | None = None
    cityName: str | None = None
    stateCode: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class Location(LooseBaseModel):
    dateTime: str | None = None
    locationCode: str | None = None
    address: Address | None = None
    name: str | None = None
    googlePlaceId: str | None = None
    uicCode: str | None = None


class TransferOffer(LooseBaseModel):
    """Transfer offer from search response."""

    type: str | None = None
    id: str | None = None
    transferType: str | None = None
    start: Location | None = None
    end: Location | None = None
    duration: str | None = None
    vehicle: Vehicle | None = None
    serviceProvider: ServiceProvider | None = None
    quotation: Quotation | None = None
    converted: Quotation | None = None
    cancellationRules: list[CancellationRule] | None = None
    methodsOfPaymentAccepted: list[str] | None = None
    language: str | None = None


class TransferSearchResponse(LooseBaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)  # type: ignore


# ============================================================
# Booking Models
# ============================================================

class Contact(LooseBaseModel):
    """Contact information for passenger."""

    phoneNumber: str | None = None
    email: str | None = None


class AddressCommon(LooseBaseModel):
    """Common address format for billing."""

    line: str | None = None
    zip: str | None = None
    countryCode: str | None = None
    cityName: str | None = None
    stateCode: str | None = None


class Passenger(LooseBaseModel):
    """Passenger data for booking."""

    firstName: str
    lastName: str
    title: str | None = None  # MR, MRS, MS, etc.
    contacts: Contact | None = None
    billingAddress: AddressCommon | None = None


class CreditCard(LooseBaseModel):
    """Credit card payment information."""

    number: str  # 16 digit card number
    holderName: str
    vendorCode: str  # VI, CA, AX, DC, etc.
    expiryDate: str  # MMYY format (e.g., "1027" for Oct 2027)
    cvv: str | None = None  # 3-4 digit CVV


class Payment(LooseBaseModel):
    """Payment information for booking."""

    methodOfPayment: MethodOfPayment
    creditCard: CreditCard | None = None
    paymentReference: str | None = None
    paymentServiceProvider: Literal["STRIPE_CONNECT"] | None = None


class Equipment(LooseBaseModel):
    """Extra equipment for booking."""

    code: str
    itemId: str | None = None


class ExtraService(LooseBaseModel):
    """Extra service for booking."""

    code: str
    itemId: str | None = None


class TransferBookingRequest(LooseBaseModel):
    """Request model for booking a transfer."""

    offerId: str
    passengers: list[Passenger]
    payment: Payment
    note: str | None = None
    flightNumber: str | None = None
    equipment: list[Equipment] | None = None
    extraServices: list[ExtraService] | None = None
    startConnectedSegment: TravelSegment | None = None
    endConnectedSegment: TravelSegment | None = None

    def to_api(self) -> dict[str, Any]:
        """Convert request to Amadeus API payload."""
        data: dict[str, Any] = {
            "passengers": [],
            "payment": self.payment.model_dump(exclude_none=True),
        }

        # Add passengers
        for p in self.passengers:
            passenger_data: dict[str, Any] = {
                "firstName": p.firstName,
                "lastName": p.lastName,
            }
            if p.title:
                passenger_data["title"] = p.title
            if p.contacts:
                passenger_data["contacts"] = p.contacts.model_dump(
                    exclude_none=True)
            if p.billingAddress:
                passenger_data["billingAddress"] = p.billingAddress.model_dump(
                    exclude_none=True)
            data["passengers"].append(passenger_data)

        # Add optional fields
        if self.note:
            data["note"] = self.note
        if self.flightNumber:
            data["flightNumber"] = self.flightNumber
        if self.equipment:
            data["equipment"] = [
                e.model_dump(exclude_none=True) for e in self.equipment
            ]
        if self.extraServices:
            data["extraServices"] = [
                s.model_dump(exclude_none=True) for s in self.extraServices
            ]
        if self.startConnectedSegment:
            data["startConnectedSegment"] = self.startConnectedSegment.model_dump(
                exclude_none=True)
        if self.endConnectedSegment:
            data["endConnectedSegment"] = self.endConnectedSegment.model_dump(
                exclude_none=True)

        return {"data": data}


class TransferReservation(LooseBaseModel):
    """Transfer reservation in order response."""

    confirmNbr: str | None = None
    status: str | None = None  # CONFIRMED, CANCELLED
    note: str | None = None
    methodOfPayment: str | None = None
    offerId: str | None = None
    transferType: str | None = None
    start: Location | None = None
    end: Location | None = None
    vehicle: Vehicle | None = None
    serviceProvider: ServiceProvider | None = None
    quotation: Quotation | None = None


class TransferOrder(LooseBaseModel):
    """Transfer order response."""

    type: str | None = None
    id: str | None = None
    reference: str | None = None
    transfers: list[TransferReservation] | None = None
    passengers: list[Passenger] | None = None


class ApiError(LooseBaseModel):
    """API error structure."""

    code: int | None = None
    title: str | None = None
    detail: str | None = None
    status: int | None = None


class TransferBookingResponse(LooseBaseModel):
    """Response from booking API."""

    data: TransferOrder | dict[str, Any] | None = None
    errors: list[ApiError] | list[dict[str, Any]] | None = None

    @property
    def is_success(self) -> bool:
        """Check if the booking was successful."""
        return self.data is not None and not self.errors


# ============================================================
# Cancellation Models
# ============================================================

class TransferCancellation(LooseBaseModel):
    """Cancellation response data."""

    confirmNbr: str | None = None
    reservationStatus: str | None = None  # CANCELLED, CONFIRMED


class TransferCancellationResponse(LooseBaseModel):
    """Response from cancellation API."""

    data: TransferCancellation | dict[str, Any] | None = None
    errors: list[ApiError] | list[dict[str, Any]] | None = None


# ============================================================
# Async Client with Caching
# ============================================================

class _TTLCache:
    """Simple in-memory TTL cache for search responses."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


class AmadeusTransferAsyncClient:
    """
    Fully async, typed Amadeus Transfer SDK client.

    Features:
    - Async HTTP (httpx)
    - Typed input & output
    - OAuth token auto-refresh
    - Retry with exponential backoff
    - In-memory TTL cache for searches
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://test.api.amadeus.com/v1",
        retries: int = 3,
        timeout: int = 15,
        cache_ttl: int = 300,
    ) -> None:
        self.api_key = api_key or os.getenv("AMADEUS_API_KEY")
        self.api_secret = api_secret or os.getenv("AMADEUS_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("AMADEUS_API_KEY and AMADEUS_API_SECRET required")

        self.base_url = base_url
        self.retries = retries
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cache = _TTLCache(cache_ttl)
        self.token: str | None = None

    async def _get_token(self) -> str:
        token_file = ".amadeus_token"

        # Try to load from file
        if os.path.exists(token_file):
            try:
                with open(token_file) as f:
                    self.token = f.read().strip()
                logger.info("Loaded token from cache")
                return self.token
            except Exception as e:
                logger.warning("Failed to load token from file: %s", e)

        # Fetch new token
        logger.info("Fetching OAuth token")
        response = await self._client.post(
            f"{self.base_url}/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
        if not self.token:
            raise RuntimeError("Failed to obtain OAuth token")

        # Save to file
        try:
            with open(token_file, "w") as f:
                f.write(self.token)
        except Exception as e:
            logger.warning("Failed to save token to file: %s", e)

        return self.token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.token:
            await self._get_token()

        for attempt in range(1, self.retries + 1):
            try:
                logger.info("%s %s (attempt %d)", method, path, attempt)

                response = await self._client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Accept": "application/vnd.amadeus+json",
                        "Content-Type": "application/json",
                    },
                    json=json,
                    params=params,
                )

                if response.status_code == 401:
                    logger.warning("Token expired, refreshing")
                    await self._get_token()
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as exc:
                logger.error("Request failed: %s", exc)
                if attempt == self.retries:
                    raise
                await httpx.AsyncClient().aclose()
                time.sleep(2 ** attempt)

        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    async def search_transfers(
        self, request: TransferSearchRequest
    ) -> TransferSearchResponse:
        cache_key = request.model_dump_json()
        cached = self._cache.get(cache_key)
        if cached:
            logger.info("Returning cached transfer search result")
            return cached

        raw = await self._request(
            "POST",
            "/shopping/transfer-offers",
            json=request.to_api(),
        )

        parsed = TransferSearchResponse.model_validate(raw)
        self._cache.set(cache_key, parsed)
        return parsed

    async def book_transfer(
        self, request: TransferBookingRequest
    ) -> TransferBookingResponse:
        print("\n\n\n", request.to_api(), "\n\n\n")
        raw = await self._request(
            "POST",
            "/ordering/transfer-orders",
            json=request.to_api(),
            params={"offerId": request.offerId},
        )
        return TransferBookingResponse.model_validate(raw)

    async def cancel_transfer(
        self, order_id: str, confirm_nbr: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/ordering/transfer-orders/{order_id}/transfers/cancellation",
            params={"confirmNbr": confirm_nbr},
        )

    async def close(self) -> None:
        """Close underlying HTTP client."""
        await self._client.aclose()


if __name__ == "__main__":
    import asyncio
    from datetime import timedelta

    from dotenv import load_dotenv

    load_dotenv()

    async def main() -> None:
        client = AmadeusTransferAsyncClient()

        # ----------------------------
        # 1️⃣ Search transfers
        # ----------------------------
        # request = TransferSearchRequest(
        #     startDateTime=datetime.now(UTC) + timedelta(days=3),
        #     startLocationCode="CDG",
        #     endAddressLine="5 Avenue Anatole France",
        #     endCityName="Paris",
        #     endZipCode="75007",
        #     endCountryCode="FR",
        #     endGeoCode=GeoCode(latitude=48.858093, longitude=2.294694),
        #     transferType="PRIVATE",
        # )

        # offers = await client.search_transfers(request)

        offer: dict[str, Any] = {
            "id": "7068429332",
            "type": "transfer-offer",
            "language": "EN",
            "transferType": "PRIVATE",
            "start": {
                "dateTime": "2025-12-18T13:57:39",
                "locationCode": "CDG"
            },
            "end": {
                "dateTime": "2025-12-18T14:57:39",
                "address": {
                    "line": "5 Avenue Anatole France",
                    "zip": "75007",
                    "countryCode": "FR",
                    "cityName": "Paris",
                    "latitude": 48.858093,
                    "longitude": 2.294694
                }
            },
            "vehicle": {
                "code": "VAN",
                "category": "ST",
                "description": "Toyota Innova or similar",
                "imageURL": "https://oss.heycars.cn/vehicle/v2/toyoto-Innova-7.png",
                "baggages": [
                    {
                        "count": 4,
                        "size": "M"
                    }
                ],
                "seats": [
                    {
                        "count": 4
                    }
                ]
            },
            "converted": {
                "monetaryAmount": "52.63",
                "currencyCode": "EUR",
                "totalFees": {
                    "monetaryAmount": "0"
                },
                "base": {
                    "monetaryAmount": "52.63"
                }
            },
            "serviceProvider": {
                "code": "HCS",
                "name": "HeyCars",
                "termsUrl": "https://oss.heycars.cn/website/TermsOfService.html",
                "logoUrl": "https://oss.heycars.cn/website/logo1.png",
                "settings": [
                    "FLIGHT_NUMBER_REQUIRED",
                    "CORPORATION_INFO_REQUIRED"
                ]
            },
            "quotation": {
                "monetaryAmount": "52.63",
                "currencyCode": "EUR",
                "totalFees": {
                    "monetaryAmount": "0"
                },
                "base": {
                    "monetaryAmount": "52.63"
                }
            },
            "supportedPaymentInstruments": [
                {
                    "vendorCode": "VI",
                    "description": "VISA"
                },
                {
                    "vendorCode": "AX",
                    "description": "AMERICANEXPRESS"
                },
                {
                    "vendorCode": "CA",
                    "description": "MASTERCARD"
                },
                {
                    "vendorCode": "DC",
                    "description": "DINERSCLUB"
                }
            ],
            "equipment": [
                {
                    "code": "CBS",
                    "description": "Booster seat for child under 135cm or up to 12 years",
                    "quotation": {
                        "monetaryAmount": "12.92",
                        "currencyCode": "EUR"
                    },
                    "isBookable": True,
                    "taxIncluded": True,
                    "includedInTotal": False,
                    "converted": {
                        "monetaryAmount": "12.92",
                        "currencyCode": "EUR"
                    }
                },
                {
                    "code": "CST",
                    "description": "Child seat determined by weight/age of child: 4-7 years/15-30 Kg",
                    "quotation": {
                        "monetaryAmount": "12.92",
                        "currencyCode": "EUR"
                    },
                    "isBookable": True,
                    "taxIncluded": True,
                    "includedInTotal": False,
                    "converted": {
                        "monetaryAmount": "12.92",
                        "currencyCode": "EUR"
                    }
                }
            ],
            "cancellationRules": [
                {
                    "feeType": "PERCENTAGE",
                    "feeValue": "100",
                    "metricType": "HOURS",
                    "metricMin": "0",
                    "metricMax": "18",
                    "ruleDescription": "Non-refundable within 18 hours of pick-up time"
                },
                {
                    "feeType": "PERCENTAGE",
                    "feeValue": "0",
                    "metricType": "HOURS",
                    "metricMin": "18",
                    "ruleDescription": "Free cancellation up to 18 hours prior to ride"
                }
            ],
            "methodsOfPaymentAccepted": [
                "CREDIT_CARD",
                "INVOICE",
                "TRAVEL_ACCOUNT"
            ],
            "distance": {
                "value": 30,
                "unit": "KM"
            },
            "extraServices": [
                {
                    "code": "FLM",
                    "description": "Flight monitoring",
                    "quotation": {
                        "monetaryAmount": "0.00",
                        "currencyCode": "EUR"
                    },
                    "isBookable": False,
                    "taxIncluded": True,
                    "includedInTotal": True,
                    "converted": {
                        "monetaryAmount": "0.00",
                        "currencyCode": "EUR"
                    }
                }
            ]
        }

        if not offer:
            print("❌ Offer not found in search results")
            await client.close()
            return

        print("Selected offer:")
        print(offer)

        offer_id = offer.get("id")

        # Provider requires flight number (from response)
        requires_flight = (
            "FLIGHT_NUMBER_REQUIRED"
            in offer.get("serviceProvider", {}).get("settings", [])
        )

        # ----------------------------
        # 2️⃣ Book transfer
        # ----------------------------
        booking_request = TransferBookingRequest(
            offerId=offer_id,
            passengers=[
                Passenger(
                    firstName="John",
                    lastName="Doe",
                    title="MR",
                    contacts=Contact(
                        phoneNumber="+33123456789",
                        email="john.doe@example.com",
                    ),
                )
            ],
            payment=Payment(
                methodOfPayment="INVOICE"
            ),
            flightNumber="AF123" if requires_flight else None,
        )

        booking_response = await client.book_transfer(booking_request)
        print("\nBooking response:")
        print(booking_response)

        # Check for errors
        if booking_response.errors:
            print("\n❌ Booking failed with errors:")
            for err in booking_response.errors:
                if isinstance(err, dict):
                    print(
                        f"  - [{err.get('code')}] {err.get('title')}: {err.get('detail')}")
                else:
                    print(f"  - [{err.code}] {err.title}: {err.detail}")
            await client.close()
            return

        # ----------------------------
        # 3️⃣ Extract order + confirm number
        # ----------------------------
        data = booking_response.data or {}

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

        print("\nOrder ID:", order_id)
        print("Confirm Number:", confirm_nbr)

        # ----------------------------
        # 4️⃣ Cancel transfer
        # ----------------------------
        if order_id and confirm_nbr:
            cancel_response = await client.cancel_transfer(
                order_id=order_id,
                confirm_nbr=confirm_nbr,
            )
            print("\nCancel response:")
            print(cancel_response)
        else:
            print("\n❌ Unable to cancel — missing order_id or confirmNbr")

        await client.close()

    asyncio.run(main())
