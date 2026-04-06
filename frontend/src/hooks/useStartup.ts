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

    async function fetchAll() {
      const [kgResult, healthResult, ...modelResults] = await Promise.allSettled([
        api.fetchKGList(),
        api.checkHealth(),
        ...vendors.map((v) => api.fetchModels(v)),
      ]);

      if (cancelled) return;

      const kgList = kgResult.status === 'fulfilled' ? kgResult.value.kgs || [] : [];
      const health = healthResult.status === 'fulfilled' ? healthResult.value : null;

      const modelsByVendor: Record<string, string[]> = {};
      vendors.forEach((vendor, i) => {
        const result = modelResults[i];
        modelsByVendor[vendor] = result.status === 'fulfilled' ? result.value.models || [] : [];
      });

      setData({ loading: false, kgList, health, modelsByVendor });
    }

    fetchAll();
    return () => {
      cancelled = true;
    };
  }, []);

  return data;
}
