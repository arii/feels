import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

type Props = {
  readonly badge?: string;
  readonly headline: string;
  readonly detail?: string;
};

export const SceneHeader: React.FC<Props> = ({ badge, headline, detail }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const entrance = spring({
    frame,
    fps,
    config: { damping: 16, stiffness: 120 },
  });

  return (
    <div
      style={{
        opacity: entrance,
        translate: `0 ${interpolate(entrance, [0, 1], [-20, 0])}px`,
      }}
      className="pointer-events-none absolute left-16 top-12 flex max-w-[1200px] flex-col items-start"
    >
      <div className="rounded-2xl border border-slate-700/60 bg-slate-950/80 px-8 py-6 shadow-xl backdrop-blur-md">
        {badge && (
          <span className="inline-block rounded-full border border-cyan-800/40 bg-cyan-950/50 px-4 py-1 font-mono text-xl font-semibold uppercase tracking-wider text-cyan-400">
            {badge}
          </span>
        )}
        <h2 className="mt-3 text-[60px] font-bold leading-[1.1] text-white">
          {headline}
        </h2>
        {detail && (
          <p className="mt-2 text-[32px] font-medium leading-snug text-slate-300">
            {detail}
          </p>
        )}
      </div>
    </div>
  );
};
