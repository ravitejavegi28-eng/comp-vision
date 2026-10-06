import { useState } from 'react';
import { ChartNoAxesCombined, Play, ShieldAlert, Timer } from 'lucide-react';
import { number, percent } from './api.js';
import { Button, EmptyState, ErrorNotice, PageHeading, SectionHeading, useResource } from './ui.jsx';

export default function Evaluation({ productId, status, revision, busy, evaluate }) {
  const source = useResource('/api/evaluation', productId, revision, true);
  const [error, setError] = useState('');
  const report = source.data?.report;
  async function runEvaluation() { setError(''); try { await evaluate(); } catch (requestError) { setError(requestError.message); } }
  if (!status?.model?.ready) return <><PageHeading title="Evaluation" description="Measure a frozen model on held-out images." /><EmptyState icon={ChartNoAxesCombined} title="Fit a model before evaluating">Add a clean reference, fit it, then test it on samples never used for calibration.</EmptyState></>;
  return <>
    <PageHeading title="Evaluation" description="The test set stays outside every fitting and threshold decision."><Button variant="primary" icon={Play} busy={busy} onClick={runEvaluation}>Evaluate held-out test set</Button></PageHeading>
    <ErrorNotice onDismiss={() => setError('')}>{error || source.error}</ErrorNotice>
    {report ? <><div className="metrics-grid"><Metric label="Defect recall" value={percent(report.defect_recall)} detail="Review or fail, out of defective tests" tone="green" /><Metric label="False reject rate" value={percent(report.false_reject_rate)} detail="Good products classified as fail" tone="red" /><Metric label="Review rate" value={percent(report.review_rate)} detail="Images needing human decision" tone="amber" /><Metric label="Mean inference time" value={`${number(report.mean_latency_ms, 0)} ms`} detail="Excludes camera and dashboard" tone="blue" /></div><section className="evaluation-panel"><SectionHeading icon={ChartNoAxesCombined} title="Test decisions"><span className="muted small">{report.total} held-out images · {report.source === 'synthetic-demo' ? 'Synthetic demonstration data' : 'Product dataset'}</span></SectionHeading><div className="confusion-grid"><DecisionRow label="Known good" values={report.confusion.good} /><DecisionRow label="Known defective" values={report.confusion.defect} /></div><div className="evaluation-notes"><ShieldAlert size={19} /><div>{report.notes.map(note => <p key={note}>{note}</p>)}</div></div></section></> : <EmptyState icon={Timer} title="No evaluation saved yet">Run a held-out test to see recall, false rejects, review rate, and latency.</EmptyState>}
  </>;
}
function Metric({ label, value, detail, tone }) { return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>; }
function DecisionRow({ label, values }) { return <div className="decision-row"><strong>{label}</strong><span className="decision-pass">Pass <b>{values.pass}</b></span><span className="decision-review">Review <b>{values.review}</b></span><span className="decision-fail">Fail <b>{values.fail}</b></span><span>Recapture <b>{values.recapture}</b></span></div>; }
