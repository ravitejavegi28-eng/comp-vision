import { useState } from 'react';
import { Archive, Database, FolderUp, Play, Plus, Upload } from 'lucide-react';
import { request } from './api.js';
import { Button, ErrorNotice, PageHeading, SectionHeading } from './ui.jsx';

export default function Dataset({ productId, status, refresh, demo, busy, train, onProductCreated }) {
  const [files, setFiles] = useState([]);
  const [split, setSplit] = useState('train');
  const [label, setLabel] = useState('good');
  const [session, setSession] = useState('capture-01');
  const [newProduct, setNewProduct] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [creating, setCreating] = useState(false);
  const counts = status?.dataset || {};
  const uploadBlocked = demo;

  async function createProduct(event) {
    event.preventDefault();
    setCreating(true); setError('');
    try {
      const product = await request('/api/products', { method: 'POST', json: { name: newProduct } });
      setNewProduct(''); onProductCreated(product);
    } catch (requestError) { setError(requestError.message); }
    finally { setCreating(false); }
  }
  async function upload(event) {
    event.preventDefault();
    if (!files.length) { setError('Choose one or more images first.'); return; }
    setUploading(true); setError(''); setMessage('');
    const body = new FormData();
    files.forEach(file => body.append('files', file));
    body.append('split', split); body.append('label', label); body.append('session_id', session);
    try {
      const result = await request('/api/dataset', { method: 'POST', productId, body });
      setMessage(result.message); setFiles([]); refresh();
    } catch (requestError) { setError(requestError.message); }
    finally { setUploading(false); }
  }
  function changeSplit(value) { setSplit(value); if (value === 'train') setLabel('good'); }
  return <>
    <PageHeading title="Dataset & model" description="Build a clean normal reference, then lock validation thresholds before testing." />
    <ErrorNotice onDismiss={() => setError('')}>{error}</ErrorNotice>
    <div className="data-layout">
      <section className="data-panel model-panel">
        <SectionHeading icon={Database} title="Current reference" />
        <div className="data-counts">
          <Count label="Training · good" value={counts.train_good || 0} target="5+" />
          <Count label="Validation · good" value={counts.validation_good || 0} target="2+" />
          <Count label="Validation · defect" value={counts.validation_defect || 0} target="2+" />
          <Count label="Test · good" value={counts.test_good || 0} target="—" />
          <Count label="Test · defect" value={counts.test_defect || 0} target="—" />
        </div>
        <div className="model-explainer"><Archive size={18} /><div><strong>{status?.model?.ready ? 'Active model is calibrated' : 'No active model'}</strong><p>{status?.model?.ready ? `${status.model.normal_count} normal reference images. Thresholds were calibrated only from validation samples.` : 'A model is never fitted from test images.'}</p></div></div>
        <Button variant="primary" className="full-width" icon={Play} busy={busy} onClick={train} disabled={!demo && (counts.train_good < 5 || counts.validation_good < 2 || counts.validation_defect < 2)}>Fit and calibrate model</Button>
        {demo ? <p className="small muted model-note">The demo’s generated samples are preloaded. Create your own product to add real camera data.</p> : <p className="small muted model-note">Training uses good images only. Validation sets thresholds. Test images stay untouched until evaluation.</p>}
      </section>
      <section className="data-panel upload-panel">
        <SectionHeading icon={FolderUp} title={uploadBlocked ? 'Start your own product' : 'Add dataset images'} />
        {uploadBlocked ? <form className="new-product" onSubmit={createProduct}><p>Synthetic samples are isolated so your real product data stays clean. Create a product workspace first.</p><label htmlFor="new-product">Product name</label><input id="new-product" value={newProduct} onChange={event => setNewProduct(event.target.value)} maxLength={80} placeholder="e.g. Granola label · version A" required /><Button variant="primary" icon={Plus} busy={creating} type="submit">Create product workspace</Button></form> : <form className="upload-form" onSubmit={upload}>
          <label className="file-drop"><Upload size={25} /><strong>{files.length ? `${files.length} image${files.length > 1 ? 's' : ''} selected` : 'Choose product images'}</strong><span>PNG, JPG, WebP or BMP · max 10 MB each</span><input type="file" accept="image/*" multiple onChange={event => setFiles(Array.from(event.target.files))} /></label>
          <div className="form-grid"><label>Dataset split<select value={split} onChange={event => changeSplit(event.target.value)}><option value="train">Training reference</option><option value="validation">Validation</option><option value="test">Held-out test</option></select></label><label>Image label<select value={label} disabled={split === 'train'} onChange={event => setLabel(event.target.value)}><option value="good">Good product</option><option value="defect">Known defect</option></select></label></div>
          <label>Capture session <input value={session} onChange={event => setSession(event.target.value)} maxLength={80} required /><span className="field-help">Keep every capture session in one split. Never split nearby video frames across training and test.</span></label>
          <Button variant="primary" icon={Upload} busy={uploading} type="submit">Add to {split}</Button>{message ? <p className="success-message">{message}</p> : null}
        </form>}
      </section>
    </div>
    <section className="dataset-rules"><h2>Data rules that protect the result</h2><div><p><strong>Normal reference</strong>Use representative good products and the same camera setup you intend to inspect.</p><p><strong>Validation</strong>Set pass and fail thresholds with separate good and defective captures.</p><p><strong>Final test</strong>Keep these images out of fitting and calibration. Include unfamiliar defect types.</p></div></section>
  </>;
}
function Count({ label, value, target }) { return <div className="data-count"><span>{label}</span><strong>{value}</strong><small>{target}</small></div>; }
