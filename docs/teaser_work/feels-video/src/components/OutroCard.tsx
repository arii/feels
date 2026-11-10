import {
  CanvasImage,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const OutroCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame: frame - 6,
    fps,
    config: { damping: 14, stiffness: 120 },
  });

  const kenBurns = interpolate(frame, [0, 5 * fps], [1, 1.04], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div className="relative h-full w-full overflow-hidden bg-slate-950">
      <CanvasImage
        src={staticFile("team.webp")}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          scale: `${kenBurns}`,
        }}
      />
      <div className="absolute inset-0 bg-slate-950/70" />

      <div
        style={{
          opacity: entrance,
          translate: `0 ${interpolate(entrance, [0, 1], [30, 0])}px`,
        }}
        className="absolute inset-0 flex items-center justify-center"
      >
        <div className="mx-24 max-w-[1100px] rounded-2xl border border-slate-700/60 bg-slate-950/80 px-12 py-10 text-center shadow-2xl backdrop-blur-md">
          <h1 className="text-7xl font-bold leading-tight text-white">
            Feels: Proprioceptive Cup Dynamics
          </h1>
          <p className="mt-5 text-4xl text-slate-200">
            Sree Dhyuti Nimmagadda &amp; Ariel Anders
          </p>
          <p className="mt-6 font-mono text-3xl">
            <span className="text-slate-400">Writeup &amp; Code: </span>
            <span className="text-cyan-300">arii.github.io/feels</span>
          </p>
          <p className="mt-2 font-mono text-2xl">
            <span className="text-slate-400">Dataset: </span>
            <span className="text-cyan-300">
              huggingface.co/datasets/dhyuti-n/feels-cup-dynamics
            </span>
          </p>
          <p className="mt-6 text-2xl text-slate-400">
            Ariel Anders Consulting
          </p>
        </div>
      </div>
    </div>
  );
};
