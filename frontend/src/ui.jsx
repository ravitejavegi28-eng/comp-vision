import { useEffect, useState } from 'react';
import { AlertCircle, LoaderCircle, X, FileImage } from 'lucide-react';
import { isWorking, request } from './api.js';
export function useResource(path, productId, revision = 0, poll = false, query = '') {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    let timer;
    setLoading(true); setError('');
    async function load() {
      try {
        const value = await request(path, { productId, signal: controller.signal, params: Object.fromEntries(new URLSearchParams(query)) });
        if (!controller.signal.aborted) { setData(value); setError(''); }
      } catch (err) { if (err.name !== 'AbortError') setError(err.message); }
      finally { if (!controller.signal.aborted) { setLoading(false); if (poll) timer = window.setTimeout(load, 3000); } }
    }
    load();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [path, productId, revision, poll, query]);
  return { data, error, loading };
}
export function Button({ children, variant = 'secondary', className = '', busy = false, icon: Icon, ...props }) {
  return <button className={`button ${variant} ${className}`} {...props} disabled={props.disabled || busy}>{busy ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : Icon ? <Icon size={16} aria-hidden="true" /> : null}{children}</button>;
}
export function ErrorNotice({ children, onDismiss, onRetry }) {
  if (!children) return null;
  return <div className="notice error" role="alert"><AlertCircle size={19} /><span>{children}</span>{onRetry ? <button onClick={onRetry}>Retry</button> : null}{onDismiss ? <button className="icon-button" aria-label="Dismiss error" onClick={onDismiss}><X size={16} /></button> : null}</div>;
}
export const DECISIONS = { pass: 'Pass', review: 'Review', fail: 'Fail', recapture: 'Recapture' };
export const SYNTHETIC = 'synthetic-demo';
export function Badge({ decision, children, className = '' }) {
  return <span className={`badge ${decision || 'neutral'} ${className}`}><span className="badge-dot" />{children || DECISIONS[decision] || decision}</span>;
}
export function EmptyState({ icon: Icon = FileImage, title, children, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={28} strokeWidth={1.5} /></div><h3>{title}</h3><p>{children}</p>{action}</div>;
}
export function PageHeading({ title, description, children }) { return <div className="page-heading"><div><h1>{title}</h1><p>{description}</p></div>{children}</div>; }
export function SectionHeading({ icon: Icon, title, children }) { return <div className="section-heading"><h2>{Icon ? <Icon size={17} /> : null}{title}</h2>{children}</div>; }
export function JobNotice({ job }) {
  if (!job) return null;
  const working = isWorking(job);
  if (!working && job.status !== 'failed') return null;
  return <div className={`notice ${working ? 'info' : 'error'}`} role={working ? 'status' : 'alert'}>{working ? <LoaderCircle size={19} className="spin" /> : <AlertCircle size={19} />}<span><strong>{working ? job.type === 'evaluate' || job.type === 'evaluation' ? 'Evaluating the model' : 'Building the product model' : 'Job could not finish'}</strong><br />{job.error || job.message || 'Preparing the next step…'}</span>{working ? <span className="job-live">In progress</span> : null}</div>;
}
