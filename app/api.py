# Async, fully-typed, cached Amadeus Transfers SDK
# Python 3.11+

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
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

    # Start location options
    startLocationCode: str | None = None
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

    # Optional passenger info
    passengers: int = Field(default=1, ge=1)
    language: str = "EN"
    currency: str | None = None

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
    data: list[dict[str, Any]] = Field(default_factory=list)
    processed: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] | None = None

    @property
    def is_success(self) -> bool:
        """Check if the booking was successful."""
        return len(self.data) > 0 and not self.errors

    def only_invoice(self) -> None:
        """Filter offers to only those accepting INVOICE payment."""
        self.processed = [
            offer for offer in self.data
            if "INVOICE" in offer.get("methodsOfPaymentAccepted", [])
        ]

    def best_offer(self, count: int = 5, page: int = 1) -> None:
        """Return top N offers sorted by monetary amount."""
        def get_amount(offer: dict[str, Any]) -> float:
            try:
                return float(offer.get("quotation", {}).get("monetaryAmount", "0"))
            except (ValueError, TypeError):
                return 0.0

        sorted_offers = sorted(
            self.data,
            key=get_amount,
            reverse=False
        )
        start = (page - 1) * count
        end = start + count
        self.processed = sorted_offers[start:end]

    def to_dict(self) -> dict[str, Any]:
        information: dict[str, Any] = {}

        for offer in self.processed:

            startLocation = offer.get("start", {}).get(
                "locationCode",
                offer.get("start", {}).get("address", {}).get("line", "N/A")
                )
            endLocation = offer.get("end", {}).get(
                "locationCode",
                offer.get("end", {}).get("address", {}).get("line", "N/A")
            )

            startDateTime = offer.get("start", {}).get("dateTime", "N/A")
            offer_id = offer.get("id", "N/A")
            quotation = offer.get("quotation", {})
            vehicle = offer.get("vehicle", {})
            provider = offer.get("serviceProvider", {})
            cancellation = offer.get("cancellationRules", [])

            information[offer_id] = {
                "startDateTime": startDateTime,
                "startLocation": startLocation,
                "endLocation": endLocation,
                "provider": provider.get("name", "N/A"),
                "price": f"{quotation.get('monetaryAmount', 'N/A')} "
                         f"{quotation.get('currencyCode', 'N/A')}",
                "vehicle": vehicle.get("description", "N/A"),
                "image": vehicle.get("imageURL", "N/A"),
                "capacity": {
                    "seats": sum(s.get("count", 0) for s in vehicle.get("seats", [])),
                    "bags": sum(b.get("count", 0) for b in vehicle.get("baggages", [])),
                },
                "cancellation": (
                    cancellation[0].get("ruleDescription", "N/A")
                    if cancellation else "N/A"
                ),
            }

        return information

    def to_str(self) -> str:
        lines: list[str] = []

        for idx, offer in enumerate(self.processed, start=1):
            quotation = offer.get("quotation", {})
            vehicle = offer.get("vehicle", {})
            provider = offer.get("serviceProvider", {})
            cancellation = offer.get("cancellationRules", [])

            # Price
            amount = quotation.get("monetaryAmount", "N/A")
            currency = quotation.get("currencyCode", "N/A")

            # Vehicle info
            vehicle_desc = vehicle.get("description", "N/A")
            vehicle_picture = vehicle.get("imageURL", "N/A")
            seats = sum(s.get("count", 0) for s in vehicle.get("seats", []))
            bags = sum(b.get("count", 0) for b in vehicle.get("baggages", []))

            # Cancellation summary (best-effort)
            cancellation_text = "Cancellation policy varies"
            if cancellation:
                cancellation_text = cancellation[0].get(
                    "ruleDescription",
                    cancellation_text
                )

            lines.append(
                f"Option {idx}:\n"
                f"- Offer ID: {offer.get('id', 'N/A')}\n"
                f"- Provider: {provider.get('name', 'N/A')}\n"
                f"- Price: {amount} {currency}\n"
                f"- Vehicle: {vehicle_desc}\n"
                f"- Image: {vehicle_picture}\n"
                f"- Capacity: {seats} seats, {bags} bags\n"
                f"- Cancellation: {cancellation_text}"
            )

        if len(lines) == 0:
            return "No transfer offers available. Tell user to modify search criteria."
        return "\n\n".join(lines)


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

    data: dict[str, Any] = Field(default_factory=dict)
    offerId: str


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
        self.token_expires_at: float | None = None

    async def _get_token(self) -> str:
        token_file = f"{self.base_url.replace('/', '_')}.token_cache"

        # Load cached token
        if os.path.exists(token_file):
            try:
                with open(token_file) as f:
                    token, expires_at = f.read().split("|")
                    if time.time() < float(expires_at):
                        self.token = token
                        self.token_expires_at = float(expires_at)
                        logger.info("Loaded OAuth token from cache")
                        return self.token
            except Exception:
                pass  # silently refresh

        logger.info("Fetching new OAuth token")

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

        payload = response.json()
        self.token = payload["access_token"]
        self.token_expires_at = time.time() + payload["expires_in"] - 30

        try:
            with open(token_file, "w") as f:
                f.write(f"{self.token}|{self.token_expires_at}")
        except Exception as e:
            logger.warning("Failed to write token cache: %s", e)

        return self.token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not self.token or not self.token_expires_at or time.time() >= self.token_expires_at:
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
                        "Idempotency-Key": str(uuid.uuid4()),
                    },
                    json=json,
                    params=params,
                )

                if response.status_code == 401:
                    logger.warning("OAuth token expired, refreshing")
                    await self._get_token()
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as exc:
                logger.error("Request failed: %s", exc)
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    async def search_transfers(
        self, request: TransferSearchRequest
    ) -> TransferSearchResponse:
        cache_key = hash(
            str(sorted(request.model_dump(exclude_none=True).items()))
        )
        cached = self._cache.get(str(cache_key))
        if cached:
            logger.info("Returning cached transfer search result")
            return cached
        try:
            raw = await self._request(
                "POST",
                "/shopping/transfer-offers",
                json=request.to_api(),
            )
        except Exception as e:
            logger.error("Transfer search failed: %s", e)
            raise

        parsed = TransferSearchResponse.model_validate(raw)
        self._cache.set(str(cache_key), parsed)
        return parsed

    async def book_transfer(
        self, request: TransferBookingRequest
    ) -> TransferBookingResponse:
        try:
            raw = await self._request(
                "POST",
                "/ordering/transfer-orders",
                json=request.data,
                params={"offerId": request.offerId},
            )
        except Exception as e:
            logger.error("Transfer booking failed: %s", e)
            raise
        return TransferBookingResponse.model_validate(raw)

    async def cancel_transfer(
        self, order_id: str, confirm_nbr: str
    ) -> dict[str, Any]:
        try:
            return await self._request(
                "POST",
                f"/ordering/transfer-orders/{order_id}/transfers/cancellation",
                params={"confirmNbr": confirm_nbr},
            )
        except Exception as e:
            logger.error("Transfer cancellation failed: %s", e)
            raise
    async def close(self) -> None:
        """Close underlying HTTP client."""
        await self._client.aclose()

CLIENT = AmadeusTransferAsyncClient()
