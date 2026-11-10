import "./index.css";
import { Composition } from "remotion";
import { FeelsShowcase } from "./FeelsShowcase";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="FeelsShowcase"
      component={FeelsShowcase}
      durationInFrames={900}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
