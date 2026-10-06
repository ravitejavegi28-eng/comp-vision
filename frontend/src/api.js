export function endpoint(path, productId, params = {}) {
  const query = new URLSearchParams({ ...params, ...(productId ? { product_id: productId } : {}) });
  return `${path}${query.size ? `?${query}` : ''}`;
}
export async function request(path, { productId, params, json, ...options } = {}) {
  let response;
  try {
    response = await fetch(endpoint(path, productId, params), {
      ...options,
      ...(json !== undefined ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(json) } : {}),
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach the inspection service. Start the local backend, then try again.');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(item => item.msg || String(item)).join('; ') : `Request failed (${response.status}). Please try again.`);
  }
  return data;
}
export const isWorking = job => !!job && ['queued', 'pending', 'running'].includes(job.status);
export const number = (value, digits = 3) => value === null || value === undefined ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
export const percent = value => value === null || value === undefined ? '—' : `${number(value * 100, 1)}%`;
export const date = value => value ? new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
