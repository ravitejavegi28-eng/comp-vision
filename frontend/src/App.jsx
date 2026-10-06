import { useCallback, useState } from 'react';
import { ArrowRight, ChevronRight, Cpu, Database, FlaskConical, History as HistoryIcon, Layers, ScanLine, ShieldCheck, ChartNoAxesCombined } from 'lucide-react';
import { isWorking, request } from './api.js';
import { Badge, ErrorNotice, JobNotice, SYNTHETIC, useResource } from './ui.jsx';
import Inspect from './Inspect.jsx';
import Dataset from './Dataset.jsx';
import Evaluation from './Evaluation.jsx';
import History from './History.jsx';
const NAV = [{ id: 'inspect', label: 'Inspect', icon: ScanLine }, { id: 'dataset', label: 'Dataset & model', icon: Database }, { id: 'evaluation', label: 'Evaluation', icon: ChartNoAxesCombined }, { id: 'history', label: 'History', icon: HistoryIcon }];
export default function App() {
  const [tab, setTab] = useState('inspect');
  const [productId, setProductId] = useState('demo-label');
  const [revision, setRevision] = useState(0);
  const [productRevision, setProductRevision] = useState(0);
  const [actionError, setActionError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const products = useResource('/api/products', null, productRevision);
  const statusResource = useResource('/api/status', productId, revision, true);
  const status = statusResource.data?.product?.id === productId ? statusResource.data : null;
  const refresh = useCallback(() => setRevision(value => value + 1), []);
  const busy = isWorking(status?.job) || submitting;
  const source = status?.product?.source || products.data?.items?.find(item => item.id === productId)?.source;
  const demo = source === SYNTHETIC || productId === 'demo-label';
  const ready = !!status?.model?.ready;
  async function startJob(type) {
    setSubmitting(true); setActionError('');
    try { await request(`/api/${type}`, { method: 'POST', productId }); refresh(); }
    catch (err) { setActionError(err.message); }
    finally { setSubmitting(false); }
  }
  function switchProduct(id) { setProductId(id); setActionError(''); }
  function onProductCreated(product) { setProductRevision(value => value + 1); switchProduct(product.id); }
  const activeNav = NAV.find(item => item.id === tab);
  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to workspace</a>
    <aside className="sidebar">
      <a className="brand" href="#inspect" onClick={event => { event.preventDefault(); setTab('inspect'); }} aria-label="Inspectra home"><div className="brand-mark"><ScanLine size={24} strokeWidth={1.7} /></div><span>inspectra<span className="brand-caption">Visual quality inspection</span></span></a>
      <div className="workspace-label">Workspace</div>
      <nav aria-label="Main navigation">{NAV.map(item => <button key={item.id} className={`nav-item ${tab === item.id ? 'active' : ''}`} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}><item.icon size={19} /><span>{item.label}</span>{tab === item.id ? <ChevronRight size={15} className="nav-arrow" /> : null}</button>)}</nav>
      <div className="sidebar-guide"><div className="guide-icon"><Layers size={19} /></div><strong>Learn what good looks like.</strong><p>Inspect every sample against your product’s normal reference.</p><button onClick={() => setTab('dataset')}>Manage reference <ArrowRight size={14} /></button></div>
      <div className="sidebar-footer"><ShieldCheck size={17} /><div>Local workspace<span>Images stay on this computer</span></div></div>
    </aside>
    <div className="workspace"><header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={14} /><strong>{activeNav.label}</strong></div><div className="engine-status"><span className={`connection-dot ${statusResource.error ? 'offline' : status ? 'online' : ''}`} />{statusResource.error ? 'Service unavailable' : status ? 'Local engine connected' : 'Connecting to engine'}<span className="device-chip"><Cpu size={13} />{status?.device || 'Local'}</span></div></header>
      <main id="main-content" className="main-content">
        <div className="product-bar"><div className="product-picker"><label htmlFor="active-product">Product</label><select id="active-product" value={productId} onChange={event => switchProduct(event.target.value)} disabled={submitting}>{products.data?.items?.length ? products.data.items.map(item => <option key={item.id} value={item.id}>{item.name}</option>) : <option value="demo-label">Packaging label demo</option>}</select></div><div className="product-meta">{demo ? <span className="source-badge"><FlaskConical size={14} />Synthetic demo data</span> : <span className="source-badge own"><Database size={14} />Your product data</span>}<Badge decision={ready ? 'pass' : 'neutral'}>{ready ? 'Model ready' : 'Model not fitted'}</Badge></div></div>
        <ErrorNotice onRetry={refresh}>{statusResource.error || products.error}</ErrorNotice><ErrorNotice onDismiss={() => setActionError('')}>{actionError}</ErrorNotice><JobNotice job={status?.job} />
        {tab === 'inspect' ? <Inspect key={productId} productId={productId} status={status} revision={revision} refresh={refresh} demo={demo} busy={busy} train={() => startJob('train')} openDataset={() => setTab('dataset')} /> : null}
        {tab === 'dataset' ? <Dataset key={productId} productId={productId} status={status} refresh={refresh} demo={demo} busy={busy} train={() => startJob('train')} onProductCreated={onProductCreated} /> : null}
        {tab === 'evaluation' ? <Evaluation key={productId} productId={productId} status={status} revision={revision} busy={busy} evaluate={() => startJob('evaluate')} /> : null}
        {tab === 'history' ? <History key={productId} productId={productId} revision={revision} refresh={refresh} /> : null}
        <footer className="page-footer"><span>Inspectra <span className="footer-divider">/</span> Local inspection workstation</span><span>{status?.app_version ? `v${status.app_version}` : 'PatchCore visual inspection'}<span className="footer-dot" />Human review supported</span></footer>
      </main>
    </div>
  </div>;
}
