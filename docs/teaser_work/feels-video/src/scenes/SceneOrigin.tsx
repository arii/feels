import { AbsoluteFill, staticFile } from "remotion";
import { Video } from "@remotion/media";
import { SceneHeader } from "../components/SceneHeader";

export const SceneOrigin: React.FC = () => {
  return (
    <AbsoluteFill className="bg-slate-950">
      <Video
        name="Arm approach"
        src={staticFile("swirl.mp4")}
        objectFit="cover"
        style={{ width: "100%", height: "100%" }}
      />
      <SceneHeader
        badge="Viam Physical AI Hackathon"
        headline="Teaching a robot to feel inside a sealed cup"
        detail="Ariel Anders & Sree Dhyuti Nimmagadda"
      />
    </AbsoluteFill>
  );
};
