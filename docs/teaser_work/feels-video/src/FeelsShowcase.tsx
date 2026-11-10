import { AbsoluteFill, useVideoConfig } from "remotion";
import { TransitionSeries } from "@remotion/transitions";
import { SceneOrigin } from "./scenes/SceneOrigin";
import { SceneMotion } from "./scenes/SceneMotion";
import { SceneTorques } from "./scenes/SceneTorques";
import { SceneValidation } from "./scenes/SceneValidation";
import { SceneOutro } from "./scenes/SceneOutro";

export const FeelsShowcase: React.FC = () => {
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill className="bg-slate-950">
      <TransitionSeries name="Feels teaser">
        <TransitionSeries.Sequence
          name="Origin"
          durationInFrames={6 * fps}
          trimBefore={1 * fps}
        >
          <SceneOrigin />
        </TransitionSeries.Sequence>

        <TransitionSeries.Sequence
          name="Motion"
          durationInFrames={7 * fps}
          trimBefore={13 * fps}
        >
          <SceneMotion />
        </TransitionSeries.Sequence>

        <TransitionSeries.Sequence
          name="Torques"
          durationInFrames={7 * fps}
          trimBefore={20 * fps}
        >
          <SceneTorques />
        </TransitionSeries.Sequence>

        <TransitionSeries.Sequence
          name="Validation"
          durationInFrames={5 * fps}
          trimBefore={30 * fps}
        >
          <SceneValidation />
        </TransitionSeries.Sequence>

        <TransitionSeries.Sequence name="Outro" durationInFrames={5 * fps}>
          <SceneOutro />
        </TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
