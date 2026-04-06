import type { KGListItem } from '../../types/app';

interface KGSelectorProps {
  kgList: KGListItem[];
  selectedKG: string;
  onSelectKG: (value: string) => void;
}

export function KGSelector({ kgList, selectedKG, onSelectKG }: KGSelectorProps) {
  return (
    <fieldset className="fieldset">
      <legend className="fieldset-legend text-xs">KG Name</legend>
      <select className="select select-bordered select-sm w-full" value={selectedKG} onChange={(e) => onSelectKG(e.target.value)}>
        <option value="">-- New KG --</option>
        {kgList.map((kg) => (
          <option key={kg.name} value={kg.name}>{kg.name}</option>
        ))}
      </select>
    </fieldset>
  );
}
