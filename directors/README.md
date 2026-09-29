# Director Architecture v1

The pipeline is now split into two responsibilities:

```text
AI / vision / audio intelligence
             |
             v
         DIRECTOR
             |
       DirectorPlan JSON
             |
             v
          RENDERER
             |
             v
       Final Short
```

## Contract

`DirectorAction` contains:
- `start`, `end`: source-video timestamps
- `action`: renderer instruction, e.g. `reaction_takeover`, `speaker_focus`
- `reason`: why the director chose it
- `priority`: conflict resolution
- `params`: renderer-specific parameters

`DirectorPlan` contains:
- `version`
- `genre`
- `confidence`
- `actions`
- `metadata`

## Current directors

### Streamer
Uses the existing camera + face intelligence and emits `reaction_takeover` actions.
The action is static for the event duration. There is explicitly no continuous zoom.

### Podcast / interview
The contract is ready, but speaker identity is intentionally not guessed yet.
The next intelligence milestone is speaker diarization + speaker-to-face association.

### Universal
Conservative fallback that preserves the source framing until a specialized director
has enough evidence to make a better decision.

## Integration order

1. `cut.py` reads `DirectorPlan`.
2. `cut.py` stops deciding *why* a shot should happen.
3. Renderer executes only the actions in the plan.
4. Add speaker diarization.
5. Add podcast/interview actions.
6. Add screen/gameplay/slide directors.
7. Add a learned/LLM director above the specialist directors.
