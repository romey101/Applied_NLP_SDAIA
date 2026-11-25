export const PUBLIC_DEMO = true;
let snapshot: Promise<any> | undefined;
export async function demoApi(path: string, body?: object) {
  if (body) throw new Error('Live AI requires the local backend.');
  snapshot ??= fetch(import.meta.env.BASE_URL + 'demo.json').then(r => {
    if (!r.ok) throw new Error('Demo data unavailable');
    return r.json();
  }).catch(e => { snapshot = undefined; throw e; });
  const data = await snapshot;
  const key = path.split('?')[0].slice(1);
  if (!(key in data)) throw new Error('Not available in public demo');
  return data[key];
}
