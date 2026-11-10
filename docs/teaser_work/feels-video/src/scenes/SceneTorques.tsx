import { AbsoluteFill, staticFile } from "remotion";
import { Video } from "@remotion/media";
import { SceneHeader } from "../components/SceneHeader";

export const SceneTorques: React.FC = () => {
  return (
    <AbsoluteFill className="bg-slate-950">
      <Video
        name="Torque telemetry"
        src={staticFile("screen.mp4")}
        objectFit="cover"
        style={{ width: "100%", height: "100%" }}
      />
      <SceneHeader
        badge="Joint Torque Residuals"
        headline="Comparing actual motor torques against a nominal baseline"
      />
    </AbsoluteFill>
  );
};
