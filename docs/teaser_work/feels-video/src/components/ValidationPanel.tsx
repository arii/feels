import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const ValidationPanel: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({ frame, fps, config: { damping: 15 } });

  return (
    <div
      style={{
        opacity: progress,
        translate: `${interpolate(progress, [0, 1], [40, 0])}px 0`,
      }}
      className="absolute right-16 top-28 w-[620px] rounded-2xl border border-slate-700/70 bg-slate-950/85 p-10 text-left shadow-2xl backdrop-blur-md"
    >
      <span className="inline-block rounded-full border border-emerald-800/40 bg-emerald-950/60 px-4 py-1 font-mono text-xl font-semibold uppercase tracking-wider text-emerald-400">
        21 Held-Out Test Cups
      </span>

      <div className="mt-5">
        <div className="text-[96px] font-extrabold leading-none text-white">
          86–89%
        </div>
        <div className="mt-1 text-3xl font-semibold text-slate-200">
          Empty vs. Non-empty
        </div>
      </div>

      <div className="mt-6 border-t border-slate-700/60 pt-6">
        <div className="text-[64px] font-extrabold leading-none text-white">
          52%
        </div>
        <div className="mt-1 text-3xl font-semibold text-slate-200">
          3-Way (Empty / Liquid / Solid)
        </div>
      </div>

      <p className="mt-6 text-2xl leading-relaxed text-slate-400">
        Nearest-neighbor baseline on a hackathon dataset
      </p>
    </div>
  );
};
