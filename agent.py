from datetime import datetime
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)


def create_agent() -> Agent:
    agent = Agent(
        name="ASI-agent",
        seed="agent_seedphrase",
        port=8001,
        mailbox=True,
        publish_agent_details=True,
    )

    protocol = Protocol(spec=chat_protocol_spec)

    @protocol.on_message(ChatMessage)
    async def handle_message(ctx: Context, sender: str, msg: ChatMessage) -> None:
        # send the acknowledgement for receiving the message
        await ctx.send(
            sender,
            ChatAcknowledgement(timestamp=datetime.now(),
                                acknowledged_msg_id=msg.msg_id),
        )

        # collect up all the text chunks
        text = ''
        for item in msg.content:
            if isinstance(item, TextContent):
                text += item.text

        # generate a response (echo in this case)
        response = f"Echo: {text}"

        # send the response back to the user
        await ctx.send(sender, ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[
                TextContent(type="text", text=response)
            ]
        ))

    @protocol.on_message(ChatAcknowledgement)
    async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
        pass

    # attach the protocol to the agent
    agent.include(protocol, publish_manifest=True)

    return agent
