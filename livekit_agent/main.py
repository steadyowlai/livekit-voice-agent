import logging
import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    MetricsCollectedEvent,
    inference,
    llm,
    metrics,
    room_io,
    stt,
    tts,
)
from livekit.agents.llm.mcp import MCPServerHTTP, MCPToolset
from livekit.agents import TurnHandlingOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit_agent.assistant import Assistant

logger = logging.getLogger(__name__)

load_dotenv()

server = AgentServer()


# The entrypoint function runs when a participant joins the room
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Configure the voice pipeline with STT, LLM, TTS, VAD, and MCP providers
    session = AgentSession(
        # LLM with fallback: OpenAI primary, Gemini backup
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash"),
            ]
        ),
        # STT with fallback: Deepgram Flux primary, Nova-3 backup
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("deepgram/flux-general-en"),
                inference.STT.from_model_string("deepgram/nova-3"),
            ]
        ),
        # TTS with fallback: Cartesia primary, Inworld backup
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string(
                    "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
                ),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=silero.VAD.load(),
        # Flux fires end-of-turn signals natively; turn_detection="stt" delegates to it.
        # NOTE: If Flux fails over to Nova-3, Nova-3 does NOT emit native end-of-turn signals.
        # In that case, swap these two lines: comment out turn_handling and re-enable turn_detection.
        turn_handling=TurnHandlingOptions(turn_detection="stt"),
        # turn_detection=MultilingualModel(),  # Use this instead when falling back to Nova-3
        preemptive_generation=True,
        # Session-scoped MCP Toolset: survives all agent handoffs and TaskGroup transitions
        tools=[
            MCPToolset(
                id="bank_mcp",
                mcp_server=MCPServerHTTP(
                    url="http://127.0.0.1:8000/sse",
                    transport_type="sse",
                ),
            )
        ],
    )

    #########################################
    # ---- START OF METRICS COLLECTION -----#
    #########################################

    # Aggregate data across all conversation turns
    usage_collector = metrics.UsageCollector()

    # Track End of Utterance timing (when turn detector decides user finished speaking)
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        # Capture EOU metrics for TTFA calculation
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        # Log each metric as it arrives and add to usage collector
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        # Print per-session summary (tokens, audio duration, costs)
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    # Fire log_usage when worker shuts down
    ctx.add_shutdown_callback(log_usage)

    # Track end of utterance to first audio (TTFA)
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            if last_eou_metrics:
                # Calculate time since user finished speaking
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"Time to first audio: {elapsed:.3f}s")

    #########################################
    # ---- END OF METRICS COLLECTION -----#
    #########################################

    # Start the session with noise cancellation enabled
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
