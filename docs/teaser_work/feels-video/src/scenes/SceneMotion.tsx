import { AbsoluteFill, staticFile } from "remotion";
import { Video } from "@remotion/media";
import { SceneHeader } from "../components/SceneHeader";

export const SceneMotion: React.FC = () => {
  return (
    <AbsoluteFill className="bg-slate-950">
      <Video
        name="Lift and swirl"
        src={staticFile("swirl.mp4")}
        objectFit="cover"
        style={{ width: "100%", height: "100%" }}
      />
      <SceneHeader
        badge="Pipeline: Lift • Swirl • Predict"
        headline="Deterministic Swirl Motion"
        detail="6-DOF xArm executing a fixed trajectory"
      />
    </AbsoluteFill>
  );
};
