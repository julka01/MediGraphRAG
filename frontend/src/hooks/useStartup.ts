import { useEffect, useState } from 'react';
import { api } from '../api';
import type { HealthResponse, KGListItem } from '../types/app';

interface StartupData {
  loading: boolean;
  kgList: KGListItem[];
  health: HealthResponse | null;
  modelsByVendor: Record<string, string[]>;
}

export function useStartup(vendors: string[]): StartupData {
  const [data, setData] = useState<StartupData>({
    loading: true,
    kgList: [],
    health: null,
    modelsByVendor: {},
  });

  // biome-ignore lint/correctness/useExhaustiveDependencies: vendors is passed once at startup; re-fetching on array identity change is not desired
  useEffect(() => {
    let cancelled = false;

    // Fire all requests in parallel but update state as each resolves
    // so fast responses render immediately without waiting for slow ones.
    api.fetchKGList().then(
      (res) => !cancelled && setData((d) => ({ ...d, kgList: res.kgs || [] })),
      () => {},
    );

    api.checkHealth().then(
      (res) => !cancelled && setData((d) => ({ ...d, health: res })),
      () => {},
    );

    const pending = vendors.map((vendor) =>
      api.fetchModels(vendor).then(
        (res) =>
          !cancelled &&
          setData((d) => ({
            ...d,
            modelsByVendor: { ...d.modelsByVendor, [vendor]: res.models || [] },
          })),
        () => {},
      ),
    );

    // Mark loading done once everything has settled
    Promise.allSettled([...pending]).then(() => {
      if (!cancelled) setData((d) => ({ ...d, loading: false }));
    });

    return () => {
      cancelled = true;
    };
  }, []);

  return data;
}
