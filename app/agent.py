import os
from datetime import datetime
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.contrib.protocols.payment import (
    CommitPayment,
    CompletePayment,
    Funds,
    RejectPayment,
    RequestPayment,
    payment_protocol_spec,
)

from app.ai import AI
from app.fet_payments import verify_fet_payment
from app.skyfire import verify_and_charge


async def send_payment_request(
    ctx: Context,
    sender: str,
    amount_usdc: float | None,
    amount_fet: float | None,
    description: str,
    deadline_seconds: int = 300
) -> dict[str, str]:
    """
    Send a payment request to the user via uAgents payment protocol.

    Args:
        ctx: uAgents context
        sender: User address to send payment request to
        amount_usdc: Amount in USDC to request (optional)
        amount_fet: Amount in FET to request (optional)
        description: Description of what the payment is for
        deadline_seconds: How long the payment request is valid (default 5 minutes)
    """
    # Get Skyfire service ID from environment
    skyfire_service_id = os.getenv("SELLER_SERVICE_ID")
    wallet_address = "fetch1casrcthxesrsgyfy56yzfuecq49h3h76afzwg7"

    # Define accepted payment methods
    accepted_funds: list[Funds] = []
    if amount_usdc is not None:
        accepted_funds.append(
            Funds(
                currency="USDC",
                amount=f"{amount_usdc:.2f}",
                payment_method="skyfire"
            )
        )
    if amount_fet is not None:
        accepted_funds.append(
            Funds(
                currency="FET",
                amount=f"{amount_fet:.5f}",
                payment_method="fet_direct"
            )
        )

    if len(accepted_funds) == 0:
        ctx.logger.error("No accepted funds specified for payment request")
        return {"status": "error", "message": "No accepted funds specified"}

    # Build metadata for payment request
    metadata = {}
    metadata["skyfire_service_id"] = skyfire_service_id
    metadata["provider_agent_wallet"] = wallet_address

    # Create payment request
    payment_request = RequestPayment(
        accepted_funds=accepted_funds,
        recipient=wallet_address,
        deadline_seconds=deadline_seconds,
        reference=str(ctx.session),
        description=description,
        metadata=metadata,
    )

    apa = ctx.storage.get("active_payment_agents")

    apa.append((str(ctx.session), sender))

    ctx.storage.set("active_payment_agents", apa)

    # Send payment request to user
    await ctx.send(sender, payment_request)

    ctx.logger.info(
        f"Payment request sent to {sender}: {amount_usdc} USDC, "
        f"description: {description}"
    )

    return {
        "status": "sent",
        "amount_usdc": f"{amount_usdc:.2f}",
        "amount_fet": f"{amount_fet:.6f}"
    }


def create_agent() -> Agent:
    agent = Agent(
        name="ASI-agent",
        seed=os.environ.get("AMADEUS_AGENT_SEED", "asi-agent-seed-124421"),
        port=int(os.environ.get("AMADEUS_AGENT_PORT", 8000)),
        mailbox=True,
        publish_agent_details=True,
        network="testnet"
    )

    @agent.on_event("startup")
    async def on_startup(ctx: Context) -> None:
        ctx.logger.info("Agent starting up...")
        ctx.logger.info(f"Wallet address: {agent.wallet.address()}")
        ctx.storage.set("active_payment_agents", [])

    chat_protocol = Protocol(spec=chat_protocol_spec)

    @chat_protocol.on_message(ChatMessage)
    async def handle_message(ctx: Context, sender: str, msg: ChatMessage) -> None:
        """Handle incoming chat messages and process different content types."""
        try:
            session_id_str = str(ctx.session) if hasattr(
                ctx, "session") else "None"
            content_types = [type(item).__name__ for item in msg.content]
            ctx.logger.info(
                f"Chat message received, sender: {sender}, "
                f"session_id: {session_id_str}, message_id: {str(msg.msg_id)}, "
                f"content_types: {content_types}, content_count: {len(msg.content)}"
            )

            ctx.logger.info(
                f"Message received from {sender} in session: {ctx.session}"
            )

            # Send acknowledgement
            try:
                await ctx.send(
                    sender,
                    ChatAcknowledgement(
                        timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id
                    ),
                )
                ctx.logger.info(
                    f"Acknowledgement sent, sender: {sender}, message_id: {str(msg.msg_id)}"
                )
            except Exception as ack_error:
                ctx.logger.error(
                    f"Failed to send acknowledgement, sender: {sender}, "
                    f"message_id: {str(msg.msg_id)}, error: {str(ack_error)}",
                    exc_info=True,
                )

            apa = ctx.storage.get("active_payment_agents")
            if (str(ctx.session), sender) in apa:
                ctx.logger.info(
                    f"Message from {sender} in session {session_id_str} "
                    f"received during active payment. Ignoring message."
                )
                return

            # Process message content
            for i, item in enumerate(msg.content):
                try:
                    session_id_str = (
                        str(ctx.session) if hasattr(ctx, "session") else "None"
                    )
                    ctx.logger.info(
                        f"Processing message content item, sender: {sender}, "
                        f"item_index: {i}, item_type: {type(item).__name__}, "
                        f"session_id: {session_id_str}"
                    )

                    if isinstance(item, TextContent):
                        session_id_str = (
                            str(ctx.session) if hasattr(
                                ctx, "session") else "None"
                        )
                        text_preview = (
                            item.text[:100] + "..."
                            if len(item.text) > 100
                            else item.text
                        )
                        ctx.logger.info(
                            f"Processing text query, sender: {sender}, "
                            f"session_id: {session_id_str}, "
                            f"text_length: {len(item.text)}, "
                            f"text_preview: {text_preview}"
                        )
                        ctx.logger.info(
                            f"Text message from {sender}: {item.text}")

                        response = await AI.ask_agent(
                            thread_id=session_id_str,
                            agent_id=sender,
                            question=item.text
                        )

                        if response["status"] == "completed":
                            # Send the response back to the user
                            await ctx.send(sender, ChatMessage(
                                timestamp=datetime.now(),
                                msg_id=uuid4(),
                                content=[
                                    TextContent(
                                        type="text", text=response["message"]
                                    )
                                ]
                            ))
                            ctx.logger.info(
                                f"Response sent to {sender} for session_id: {session_id_str}"
                            )
                        else:
                            payment_info = response.get("payment_required")
                            ctx.storage.set(
                                f"pending_payment:{sender}:{session_id_str}",
                                {
                                    "thread_id": session_id_str,
                                    "agent_id": sender,
                                    "payment_info": payment_info,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )

                            # Send payment request
                            res = await send_payment_request(
                                ctx=ctx,
                                sender=sender,
                                amount_usdc=payment_info["amount_usdc"],
                                amount_fet=payment_info["amount_fet"],
                                description=(
                                    f"Transfer booking: "
                                    f"{payment_info['pickup_location']} → "
                                    f"{payment_info['dropoff_location']}"
                                )
                            )

                            if res.get("status") != "sent":
                                await ctx.send(sender, ChatMessage(
                                    timestamp=datetime.now(),
                                    msg_id=uuid4(),
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=(
                                                "⚠️ Failed to initiate payment request. "
                                                "Please try again later."
                                            )
                                        )
                                    ]
                                ))
                                ctx.logger.error(
                                    f"Failed to send payment request to {sender} "
                                    f"for session_id: {session_id_str}"
                                )

                            else:
                                request_overview = ""
                                request_overview += "💳 **Payment Required**\n\n"
                                request_overview += f"**Total Amount:** {payment_info['amount']} {payment_info['currency']}\n\n"
                                request_overview += f"**USDC Amount (To be paid):** {res['amount_usdc']} USDC\n\n"
                                request_overview += f"**FET Amount (To be paid):** {res['amount_fet']} FET\n\n"
                                if payment_info["paid_fet"] > 0:
                                    request_overview += f"**Already Paid:** {payment_info['paid_fet']} FET (from credits)\n\n"
                                request_overview += f"**Route:** {payment_info['pickup_location']} → "
                                request_overview += f"{payment_info['dropoff_location']}\n\n"
                                request_overview += f"**Pickup:** {payment_info['pickup_datetime']}\n\n"
                                request_overview += f"**Vehicle:** {payment_info['vehicle_type']}\n\n"
                                request_overview += f"**Provider:** {payment_info['provider']}\n\n"
                                request_overview += f"**Passengers:** {payment_info['passenger_count']}\n\n"
                                request_overview += "Please complete the payment to confirm your booking. You can pay using USDC via Skyfire or FET tokens. Thank you!"

                                await ctx.send(sender, ChatMessage(
                                    timestamp=datetime.now(),
                                    msg_id=uuid4(),
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=request_overview
                                        )
                                    ]
                                ))

                except Exception as content_error:
                    error_type = type(content_error).__name__
                    ctx.logger.error(
                        f"Error processing message content item, sender: {sender}, "
                        f"item_index: {i}, item_type: {type(item).__name__}, "
                        f"error: {str(content_error)}, error_type: {error_type}",
                        exc_info=True,
                    )
                    # Continue processing other content items
                    continue

        except Exception as e:
            msg_id = str(msg.msg_id) if hasattr(msg, "msg_id") else "None"
            ctx.logger.error(
                f"Error handling chat message, sender: {sender}, "
                f"message_id: {msg_id}, error: {str(e)}, "
                f"error_type: {type(e).__name__}",
                exc_info=True,
            )

    @chat_protocol.on_message(ChatAcknowledgement)
    async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
        pass

    payment_proto = Protocol(spec=payment_protocol_spec, role="seller")

    @payment_proto.on_message(CommitPayment)
    async def handle_payment_commit(ctx: Context, sender: str, msg: CommitPayment) -> None:
        """Handle payment commitment and verify it."""
        session_id_str = str(ctx.session)
        ctx.logger.info(
            f"Payment commit from {sender}: {msg.funds.currency} "
            f"{msg.funds.amount} via {msg.funds.payment_method}"
        )

        # Get stored payment context
        payment_ctx_key = f"pending_payment:{sender}:{session_id_str}"
        payment_ctx = ctx.storage.get(
            payment_ctx_key) if ctx.storage.has(payment_ctx_key) else None

        if not payment_ctx:
            ctx.logger.error(f"No pending payment found for {sender}")
            await ctx.send(sender, RejectPayment(reason="No pending payment found"))
            return

        # Verify payment based on method
        payment_method = msg.funds.payment_method
        verified = False

        try:
            if payment_method == "skyfire":
                ctx.logger.info("Verifying Skyfire payment...")
                verified = await verify_and_charge(
                    token=msg.transaction_id,
                    amount_usdc=str(msg.funds.amount),
                    logger=ctx.logger
                )

            elif payment_method == "fet_direct":
                ctx.logger.info("Verifying FET payment...")
                recipient = os.environ.get("FET_RECIPIENT_ADDRESS")
                if not recipient:
                    ctx.logger.error("FET_RECIPIENT_ADDRESS not configured")
                    verified = False
                else:
                    verified = await verify_fet_payment(
                        tx_hash=msg.transaction_id,
                        recipient_address=recipient,
                        amount_fet=str(msg.funds.amount),
                        logger=ctx.logger
                    )
            else:
                ctx.logger.error(
                    f"Unsupported payment method: {payment_method}")
                verified = False

        except Exception as e:
            ctx.logger.error(f"Payment verification error: {e}", exc_info=True)
            verified = False

        if verified:
            ctx.logger.info("✅ Payment verified successfully")

            # Send CompletePayment to user
            await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))

            # Notify user payment was successful
            await ctx.send(sender, ChatMessage(
                timestamp=datetime.now(),
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text="✅ Payment verified! Processing your booking..."
                    )
                ]
            ))

            # Resume LangChain agent with approval
            try:
                result = await AI.confirm_payment(
                    thread_id=payment_ctx["thread_id"],
                    agent_id=payment_ctx["agent_id"],
                    approved=True,
                    payment_ctx=payment_ctx
                )

                # Send booking result
                await ctx.send(sender, ChatMessage(
                    timestamp=datetime.now(),
                    msg_id=uuid4(),
                    content=[
                        TextContent(type="text", text=result["message"])
                    ]
                ))

                ctx.logger.info(
                    f"Booking completed for {sender} in session {session_id_str}"
                )
                ctx.logger.info(f"Response: {result['message']}")

            except Exception as e:
                ctx.logger.error(
                    f"Booking completion error: {e}", exc_info=True)
                await ctx.send(sender, ChatMessage(
                    timestamp=datetime.now(),
                    msg_id=uuid4(),
                    content=[
                        TextContent(
                            type="text",
                            text="Payment verified but booking failed, please try again."
                        )
                    ]
                ))

            # Clean up
            if ctx.storage.has(payment_ctx_key):
                ctx.storage.remove(payment_ctx_key)

        else:
            ctx.logger.error("❌ Payment verification failed")

            # Reject payment
            await ctx.send(sender, RejectPayment(
                reason="Payment verification failed. Please try again."
            ))

            result = await AI.confirm_payment(
                thread_id=payment_ctx["thread_id"],
                agent_id=payment_ctx["agent_id"],
                approved=False,
                rejection_reason="Payment verification failed",
                payment_ctx=payment_ctx
            )

            ctx.logger.info(
                f"Payment rejected for {sender} in session {session_id_str}"
            )
            ctx.logger.info(f"Response: {result['message']}")

            await ctx.send(sender, ChatMessage(
                timestamp=datetime.now(),
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=result["message"]
                    )
                ]
            ))

        apa = ctx.storage.get("active_payment_agents")
        if (str(ctx.session), sender) in apa:
            apa.remove((str(ctx.session), sender))
            ctx.storage.set("active_payment_agents", apa)

    @payment_proto.on_message(RejectPayment)
    async def handle_payment_rejection(ctx: Context, sender: str, msg: RejectPayment) -> None:
        """Handle when user rejects the payment."""
        ctx.logger.info(f"Payment rejected by {sender}: {msg.reason}")

        # Could resume agent with rejection here if needed
        # For now, just clean up
        session_id_str = str(ctx.session)
        payment_ctx_key = f"pending_payment:{sender}:{session_id_str}"
        if ctx.storage.has(payment_ctx_key):

            await ctx.send(sender, ChatMessage(
                timestamp=datetime.now(),
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text="⚠️ You have rejected the payment. Please wait while we cancel your booking."
                    )
                ]
            ))

            payment_ctx = ctx.storage.get(payment_ctx_key)

            result = await AI.confirm_payment(
                thread_id=session_id_str,
                agent_id=sender,
                approved=False,
                rejection_reason="User rejected the payment",
                payment_ctx=payment_ctx
            )

            await ctx.send(sender, ChatMessage(
                timestamp=datetime.now(),
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=result["message"]
                    )
                ]
            ))

            ctx.storage.remove(payment_ctx_key)
            apa = ctx.storage.get("active_payment_agents")
            if (str(ctx.session), sender) in apa:
                apa.remove((str(ctx.session), sender))
                ctx.storage.set("active_payment_agents", apa)

    # attach the protocol to the agent
    agent.include(chat_protocol, publish_manifest=True)
    agent.include(payment_proto, publish_manifest=True)

    return agent
