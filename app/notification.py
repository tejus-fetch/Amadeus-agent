from __future__ import annotations

import os
from pathlib import Path

import resend
from jinja2 import Environment, FileSystemLoader, select_autoescape

resend.api_key = os.environ["RESEND_API_KEY"]

def send_email(
    to: list[str],
    from_address: str,
    subject: str,
    html_content: str,
) -> resend.Emails.SendResponse:
    params: resend.Emails.SendParams = {
        "from": from_address,
        "to": to,
        "subject": subject,
        "html": html_content,
    }
    return resend.Emails.send(params)


TEMPLATE_DIR = Path(__file__).parent / "templates"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    enable_async=False,
)

def render_transfer_confirmation_email(
    *,
    customer_name: str | None,
    order_id: str,
    confirmation_number: str,
    pickup_location: str,
    dropoff_location: str,
    pickup_datetime: str,
    vehicle_type: str,
    provider: str,
    price: str,
    currency: str = "EUR",
    cancellation_policy: str,
    platform_name: str = "ASI1",
) -> tuple[str, str]:
    """
    Render transfer confirmation email HTML.
    """

    template = env.get_template("transfer_confirmation.jinja")

    subject = f"Your transfer booking confirmation - {confirmation_number}"
    body = template.render(
        customer_name=customer_name,
        order_id=order_id,
        confirmation_number=confirmation_number,
        pickup_location=pickup_location,
        dropoff_location=dropoff_location,
        pickup_datetime=pickup_datetime,
        vehicle_type=vehicle_type,
        provider=provider,
        price=price,
        currency=currency,
        cancellation_policy=cancellation_policy,
        platform_name=platform_name,
    )

    return subject, body

def render_transfer_cancellation_email(
    *,
    customer_name: str | None,
    order_id: str,
    confirmation_number: str,
    pickup_location: str,
    dropoff_location: str,
    pickup_datetime: str,
    vehicle_type: str,
    provider: str,
    refund_status: str | None = None,
    platform_name: str = "ASI1",
) -> tuple[str, str]:
    """
    Render transfer cancellation email HTML.
    """

    template = env.get_template("transfer_cancellation.jinja")

    subject = f"Your transfer booking cancellation - {confirmation_number}"
    body = template.render(
        customer_name=customer_name,
        order_id=order_id,
        confirmation_number=confirmation_number,
        pickup_location=pickup_location,
        dropoff_location=dropoff_location,
        pickup_datetime=pickup_datetime,
        vehicle_type=vehicle_type,
        provider=provider,
        refund_status=refund_status,
        platform_name=platform_name,
    )

    return subject, body
