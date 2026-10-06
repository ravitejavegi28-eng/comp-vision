import { useEffect, useState } from 'react';
import { Activity, AlertCircle, Box, Camera, CheckCircle2, ClipboardCheck, ScanLine, ShieldCheck, XCircle } from 'lucide-react';
import { date, number, request } from './api.js';
import { Badge, Button, DECISIONS, ErrorNotice, SectionHeading } from './ui.jsx';
export default function ResultPanel({ result, status, productId, onUpdated }) {
  const [editing, setEditing] = useState(false);
  const [reviewDecision, setReviewDecision] = useState('review');
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { setEditing(false); setError(''); setReviewDecision(result?.review?.decision || 'review'); setNote(result?.review?.note || ''); }, [result?.id]);
  async function saveReview(event) {
    event.preventDefault(); setSaving(true); setError('');
    try { const value = await request(`/api/inspections/${result.id}/review`, { method: 'POST', productId, json: { decision: reviewDecision, note } }); onUpdated(value); setEditing(false); }
    catch (err) { setError(err.message); }
    finally { setSaving(false); }
  }
  const title = result ? { pass: 'Within normal range', fail: 'Anomaly detected', review: 'Needs a closer look', recapture: 'Capture a clearer image' }[result.decision] : 'Ready for a closer look';
  const ResultIcon = !result ? ScanLine : result.decision === 'pass' ? CheckCircle2 : result.decision === 'fail' ? XCircle : result.decision === 'recapture' ? Camera : AlertCircle;
  return <aside className="result-panel"><SectionHeading icon={Activity} title="Inspection result">{result ? <span className="small muted">{date(result.created_at)}</span> : <span className="small muted">Awaiting image</span>}</SectionHeading>
    <div className={`result-status ${result?.decision || 'waiting'}`}><div className="result-symbol"><ResultIcon size={29} strokeWidth={1.7} /></div>{result ? <Badge decision={result.decision} /> : <span className="waiting-label">Not inspected</span>}<h2>{title}</h2><p>{result ? { pass: 'This image falls below the model’s pass threshold.', fail: 'Inspect the highlighted areas before accepting this product.', review: 'The score is between thresholds. Ask an operator to check this product.', recapture: 'Image quality prevented a reliable inspection.' }[result.decision] : 'Select a sample or capture a product to see its inspection result.'}</p></div>
    {result?.quality?.warnings?.length ? <ul className="quality-warnings">{result.quality.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul> : null}
    {result && result.score !== null ? <div className="score-section"><div className="score-label"><span>Anomaly score</span><strong>{number(result.score, 4)}</strong></div><ScoreScale score={result.score} thresholds={result.thresholds} /><p className="small muted">A model distance score, not a defect probability.</p></div> : null}
    <dl className="result-details"><div><dt>Model</dt><dd>{status?.model?.algorithm || 'PatchCore'}</dd></div><div><dt>Inspection time</dt><dd>{result?.latency_ms !== undefined ? `${number(result.latency_ms, 0)} ms` : '—'}</dd></div><div><dt>Reference images</dt><dd>{status?.model?.normal_count ?? '—'}</dd></div><div><dt>Model version</dt><dd className="truncate" title={result?.model_id || status?.model?.id}>{result?.model_id?.slice(0, 12) || status?.model?.id?.slice(0, 12) || 'Not fitted'}</dd></div></dl>
    {result?.object_detection ? <div className="yolo-panel"><div className="yolo-title"><Box size={16} /><strong>{result.object_detection.model} object check</strong><span>{result.object_detection.available ? `${result.object_detection.latency_ms} ms` : 'Unavailable'}</span></div>{result.object_detection.available ? result.object_detection.detections.length ? <ul>{result.object_detection.detections.slice(0, 4).map((item, index) => <li key={`${item.label}-${index}`}><span>{item.label}</span><strong>{number(item.confidence * 100, 0)}%</strong></li>)}</ul> : <p>No recognized COCO object in this frame. This is expected for many product labels.</p> : <p>{result.object_detection.error || 'YOLO could not run for this image.'}</p>}<small>{result.object_detection.note}</small></div> : null}
    {result ? <div className="review-section"><ErrorNotice>{error}</ErrorNotice>{result.review ? <div className="human-review"><ClipboardCheck size={17} /><div><strong>Operator decision: {DECISIONS[result.review.decision]}</strong>{result.review.note ? <p>{result.review.note}</p> : null}<span>The original model decision is preserved.</span></div></div> : null}{editing ? <form className="review-form" onSubmit={saveReview}><label htmlFor="review-decision">Operator decision</label><select id="review-decision" value={reviewDecision} onChange={event => setReviewDecision(event.target.value)}><option value="pass">Pass</option><option value="review">Needs review</option><option value="fail">Fail</option></select><label htmlFor="review-note">Inspection note <span className="muted">(optional)</span></label><textarea id="review-note" rows={2} maxLength={2000} value={note} onChange={event => setNote(event.target.value)} placeholder="What did you find?" /><div className="form-actions"><Button type="button" onClick={() => setEditing(false)}>Cancel</Button><Button type="submit" variant="primary" busy={saving}>Save review</Button></div></form> : <Button className="full-width" icon={ClipboardCheck} onClick={() => setEditing(true)}>{result.review ? 'Update operator review' : 'Add operator review'}</Button>}</div> : <div className="result-guidance"><ShieldCheck size={16} /><span>Every inspection keeps its model version and thresholds for traceability.</span></div>}
  </aside>;
}
function ScoreScale({ score, thresholds }) {
  if (!thresholds) return null;
  const max = Math.max(score * 1.15, thresholds.fail * 1.35, 0.00001);
  const pass = Math.max(0, Math.min(100, thresholds.pass / max * 100));
  const fail = Math.max(pass, Math.min(100, thresholds.fail / max * 100));
  return <><div className="score-track" aria-hidden="true"><span className="pass-range" style={{ width: `${pass}%` }} /><span className="review-range" style={{ width: `${fail - pass}%` }} /><span className="fail-range" style={{ width: `${100 - fail}%` }} /><span className="score-marker" style={{ left: `${Math.max(0, Math.min(100, score / max * 100))}%` }} /></div><div className="score-thresholds"><span>Pass &lt; {number(thresholds.pass, 4)}</span><span>Fail ≥ {number(thresholds.fail, 4)}</span></div></>;
}
