import { AbsoluteFill, staticFile } from "remotion";
import { Video } from "@remotion/media";
import { ValidationPanel } from "../components/ValidationPanel";

export const SceneValidation: React.FC = () => {
  return (
    <AbsoluteFill className="bg-slate-950">
      <Video
        name="Validation results"
        src={staticFile("screen.mp4")}
        objectFit="cover"
        style={{ width: "100%", height: "100%" }}
      />
      <div className="absolute inset-x-0 bottom-0 h-[72%] bg-linear-to-t from-slate-950/95 via-slate-950/60 to-transparent" />
      <ValidationPanel />
    </AbsoluteFill>
  );
};
