import type { ReactNode } from 'react';

import { hint, label, type TechnicalTerm } from '@son/ui-terminology';

/**
 * Renders a technical concept in business language with the technical wording
 * behind a hover hint.
 *
 * 需求方案.txt principle 4: "界面只显示业务语言，鼠标悬停显示技术说明". This is the
 * only component that knows a term is technical, so a caller passing a raw
 * identifier into visible copy is a type error rather than a review finding.
 *
 * The `?` trigger is a real button, so it is keyboard reachable and announced.
 * A `title` attribute alone would be invisible to a screen reader.
 */
export function Term({
  term,
  children,
  showHint = true,
}: {
  term: TechnicalTerm;
  /** Override the label; the hint always describes the underlying mechanism. */
  children?: ReactNode;
  showHint?: boolean;
}) {
  return (
    <span className="term">
      <span>{children ?? label(term)}</span>
      {showHint ? (
        <span className="term__hint" tabIndex={0} role="button" aria-label={`关于「${label(term)}」的说明`}>
          ?
          <span className="term__tip" role="tooltip">
            {hint(term)}
          </span>
        </span>
      ) : null}
    </span>
  );
}