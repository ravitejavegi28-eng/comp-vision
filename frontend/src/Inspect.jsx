import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Camera, Check, CircleHelp, Database, FileImage, ImagePlus, Layers, LoaderCircle, Play, ScanLine, Square, Upload } from 'lucide-react';
import { request } from './api.js';
import { Badge, Button, ErrorNotice, PageHeading, SectionHeading, useResource } from './ui.jsx';
import ResultPanel from './ResultPanel.jsx';
export default function Inspect({ productId, status, revision, refresh, demo, busy, train, openDataset }) {
  const samplesResource = useResource('/api/samples', productId, revision, false, 'split=test');
  const samples = samplesResource.data?.items || [];
  const [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState('');
  const [result, setResult] = useState(null);
  const [view, setView] = useState('original');
  const [error, setError] = useState('');
  const [inspecting, setInspecting] = useState(false);
  const [camera, setCamera] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const aliveRef = useRef(true);
  const ready = !!status?.model?.ready;
  useEffect(() => { if (!selected && !file && samples.length) setSelected(samples[0]); }, [samples, selected, file]);
  useEffect(() => {
    if (!file) { setFileUrl(''); return; }
    const url = URL.createObjectURL(file); setFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => { aliveRef.current = true; return () => { aliveRef.current = false; streamRef.current?.getTracks().forEach(track => track.stop()); }; }, []);
  useEffect(() => { if (camera && videoRef.current && streamRef.current) videoRef.current.srcObject = streamRef.current; }, [camera]);
  function stopCamera() { streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = null; setCamera(false); }
  async function startCamera() {
    setCameraStarting(true); setError('');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or a secure connection. Open this app on localhost, or upload a photo.');
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      if (!aliveRef.current) { stream.getTracks().forEach(track => track.stop()); return; }
      streamRef.current = stream; setCamera(true); setResult(null);
    } catch (err) { setError(err.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser settings, or upload an image.' : err.name === 'NotFoundError' ? 'No camera was found. Connect a camera or upload an image.' : err.message); }
    finally { if (aliveRef.current) setCameraStarting(false); }
  }
  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith('image/')) { setError('Choose a JPG, PNG, WebP, or other supported image file.'); return; }
    stopCamera(); setFile(nextFile); setSelected(null); setResult(null); setView('original'); setError('');
  }
  function capture() {
    const video = videoRef.current;
    if (!video?.videoWidth) { setError('The camera is still starting. Try capturing again in a moment.'); return; }
    const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => { if (blob && aliveRef.current) chooseFile(new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' })); }, 'image/jpeg', 0.94);
  }
  function chooseSample(sample) { stopCamera(); setSelected(sample); setFile(null); setResult(null); setView('original'); setError(''); }
  async function inspect() {
    if (!selected && !file) return;
    const body = new FormData(); if (file) body.append('file', file); else body.append('sample_id', selected.id);
    setInspecting(true); setError('');
    try {
      const value = await request('/api/inspect', { method: 'POST', productId, body });
      if (aliveRef.current) { setResult(value); setView(value.overlay_url ? 'overlay' : 'original'); refresh(); }
    } catch (err) { if (aliveRef.current) setError(err.message); }
    finally { if (aliveRef.current) setInspecting(false); }
  }
  const previewUrl = result ? view === 'overlay' ? result.overlay_url || result.image_url : view === 'heatmap' ? result.heatmap_url || result.image_url : view === 'objects' ? result.yolo_overlay_url || result.image_url : result.image_url : fileUrl || selected?.url;
  const imageName = file?.name || selected?.name || 'No image selected';
  const counts = status?.stats;
  return <>
    <PageHeading title="Inspection station" description="Capture a product. See where it differs."><div className="session-totals"><div><strong>{counts?.total ?? '—'}</strong><span>Inspected</span></div><div><strong className="text-green">{counts?.pass ?? '—'}</strong><span>Passed</span></div><div><strong className="text-red">{counts?.fail ?? '—'}</strong><span>Failed</span></div><div><strong className="text-amber">{counts?.review ?? '—'}</strong><span>To review</span></div></div></PageHeading>
    {!ready ? <div className="setup-callout"><div className="setup-icon"><Layers size={23} /></div><div><strong>{demo ? 'Fit your first inspection model' : 'Build this product’s normal reference'}</strong><p>{demo ? 'The synthetic packaging dataset is ready. Fit and calibrate the model to start inspecting.' : 'Add normal images and separate validation samples, then fit a product-specific model.'}</p></div><Button variant="primary" icon={demo ? Play : Database} busy={busy} onClick={demo ? train : openDataset}>{demo ? 'Fit demo model' : 'Set up dataset'}</Button></div> : null}
    <ErrorNotice onDismiss={() => setError('')}>{error || samplesResource.error}</ErrorNotice>
    <div className="inspection-layout">
      <section className="viewer-panel" aria-label="Inspection image">
        <div className="viewer-toolbar"><div className="viewer-source"><span className={`tiny-dot ${camera ? 'live' : ''}`} />{camera ? 'Live camera' : 'Image workspace'}</div><div className="segmented dark" aria-label="Image visualization">{['original', 'overlay', 'heatmap', 'objects'].map(mode => <button key={mode} aria-pressed={view === mode} className={view === mode ? 'selected' : ''} disabled={camera || (mode === 'objects' ? !result?.yolo_overlay_url : mode !== 'original' && !result?.heatmap_url)} onClick={() => setView(mode)}>{mode === 'original' ? 'Original' : mode === 'overlay' ? 'Overlay' : mode === 'heatmap' ? 'Heatmap' : 'Objects'}</button>)}</div></div>
        <div className={`image-stage ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); if (!inspecting) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!inspecting) chooseFile(event.dataTransfer.files[0]); }}>
          <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
          {camera ? <video ref={videoRef} autoPlay playsInline muted aria-label="Live camera preview" /> : previewUrl ? <img src={previewUrl} alt={`${view === 'original' ? 'Product image' : view === 'overlay' ? 'Anomaly overlay' : view === 'heatmap' ? 'Anomaly heatmap' : 'YOLO object overlay'}: ${imageName}`} /> : <div className="stage-empty"><ScanLine size={46} strokeWidth={1.2} /><h3>Your next inspection starts here</h3><p>Upload a product image or use your camera.</p><button onClick={() => fileInput.current?.click()}>Choose an image <ArrowRight size={16} /></button></div>}
          {inspecting ? <div className="stage-processing" role="status"><LoaderCircle size={24} className="spin" /><span>Inspecting image…</span></div> : null}
          {dragging ? <div className="drop-overlay"><ImagePlus size={28} />Drop image to inspect</div> : null}
          {previewUrl && !camera ? <div className="stage-caption"><span>{imageName}</span>{result ? <Badge decision={result.decision} /> : <span className="not-inspected">Not inspected</span>}</div> : null}
        </div>
        <div className="capture-bar"><div className="capture-actions"><input ref={fileInput} type="file" accept="image/*" className="visually-hidden" aria-label="Upload inspection image" onChange={event => { chooseFile(event.target.files[0]); event.target.value = ''; }} /><Button icon={Upload} onClick={() => fileInput.current?.click()} disabled={inspecting}>Upload image</Button>{camera ? <><Button icon={Camera} onClick={capture}>Capture</Button><button className="icon-button" onClick={stopCamera} aria-label="Stop camera"><Square size={16} /></button></> : <Button icon={Camera} onClick={startCamera} busy={cameraStarting} disabled={inspecting}>Use camera</Button>}</div><Button variant="primary" icon={ScanLine} busy={inspecting} onClick={inspect} disabled={!ready || camera || (!file && !selected)}>Inspect image</Button></div>
        <div className="capture-hint"><CircleHelp size={13} /><span>Use the same product, framing, and lighting as your reference images.</span></div>
      </section>
      <ResultPanel result={result} status={status} productId={productId} onUpdated={value => { setResult(value); refresh(); }} />
    </div>
    <section className="sample-section"><SectionHeading icon={FileImage} title={demo ? 'Try a demo sample' : 'Test samples'}><span className="muted small">{samples.length ? `${samples.length} held-out images` : 'Upload test data in Dataset & model'}</span></SectionHeading>{samples.length ? <><div className="sample-gallery">{samples.map(sample => <button key={sample.id} className={`sample-card ${selected?.id === sample.id && !file ? 'selected' : ''}`} onClick={() => chooseSample(sample)} disabled={inspecting} aria-pressed={selected?.id === sample.id && !file}><div className="sample-thumbnail"><img src={sample.url} alt={sample.name} loading="lazy" />{selected?.id === sample.id && !file ? <span className="sample-check"><Check size={12} /></span> : null}</div><span className="sample-name">{sample.name}</span><span className={`sample-label ${sample.label === 'good' ? 'good' : 'defect'}`}>Known {sample.label === 'good' ? 'good' : 'defect'}</span></button>)}</div><p className="sample-footnote">{demo ? 'Computer-generated demonstration images. Results on these samples do not establish real-world inspection accuracy.' : 'Known labels describe the dataset. The model makes its own decision when you inspect an image.'}</p></> : <div className="inline-empty">{samplesResource.loading ? 'Loading test samples…' : 'No test samples yet. You can still inspect uploaded photos.'}</div>}</section>
  </>;
}
