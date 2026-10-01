import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, LineChart } from 'echarts/charts';
import {
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsOption } from 'echarts';

echarts.use([
  BarChart,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  CanvasRenderer,
]);

/**
 * ECharts host.
 *
 * Registered components are imported individually rather than pulling all of
 * echarts: the tree-shaken bundle is a fraction of the full one, and this is a
 * fixed-cost dependency on every page.
 */
export function Chart({
  option,
  height = 240,
  label: ariaLabel,
}: {
  option: EChartsOption;
  height?: number;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    const chart = echarts.init(ref.current, undefined, { renderer: 'canvas' });
    chart.setOption(option);

    // Resize on container change: the rail collapses at 860px and the grid
    // reflows, which leaves the canvas at its old width otherwise.
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(ref.current);

    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [option]);

  return (
    <div
      ref={ref}
      className="chart"
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
    />
  );
}