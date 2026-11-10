import { AbsoluteFill } from "remotion";
import { OutroCard } from "../components/OutroCard";

export const SceneOutro: React.FC = () => {
  return (
    <AbsoluteFill className="bg-slate-950">
      <OutroCard />
    </AbsoluteFill>
  );
};
