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

from app.ai import AI


def create_agent() -> Agent:
    agent = Agent(
        name="ASI-agent",
        seed=os.environ.get("AMADEUS_AGENT_SEED", "asi-agent-seed-124421"),
        port=int(os.environ.get("AMADEUS_AGENT_PORT", 8000)),
        mailbox=True,
        publish_agent_details=True,
        network="testnet"
    )

    protocol = Protocol(spec=chat_protocol_spec)

    @protocol.on_message(ChatMessage)
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

            ctx.logger.debug(
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
                ctx.logger.debug(
                    f"Acknowledgement sent, sender: {sender}, message_id: {str(msg.msg_id)}"
                )
            except Exception as ack_error:
                ctx.logger.error(
                    f"Failed to send acknowledgement, sender: {sender}, "
                    f"message_id: {str(msg.msg_id)}, error: {str(ack_error)}",
                    exc_info=True,
                )

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
                        ctx.logger.debug(
                            f"Text message from {sender}: {item.text}")

                        response = await AI.ask_agent(
                            thread_id=session_id_str,
                            agent_id=sender,
                            question=item.text
                        )

                        # Send the response back to the user
                        await ctx.send(sender, ChatMessage(
                            timestamp=datetime.now(),
                            msg_id=uuid4(),
                            content=[
                                TextContent(type="text", text=response)
                            ]
                        ))

                    else:
                        session_id_str = (
                            str(ctx.session) if hasattr(
                                ctx, "session") else "None"
                        )
                        content_repr = (
                            str(item)[:200] + "..."
                            if len(str(item)) > 200
                            else str(item)
                        )
                        ctx.logger.warning(
                            f"Unexpected content type received, sender: {sender}, "
                            f"session_id: {session_id_str}, "
                            f"content_type: {type(item).__name__}, "
                            f"content_repr: {content_repr}"
                        )
                        ctx.logger.debug(
                            "Received unexpected content type from "
                            f"{sender}: {type(item)} - {item}"
                        )

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

    @protocol.on_message(ChatAcknowledgement)
    async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
        pass

    # attach the protocol to the agent
    agent.include(protocol, publish_manifest=True)

    return agent
