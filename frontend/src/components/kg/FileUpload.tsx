import { useRef, useState } from 'react';
import { showError } from '../ui/Notifications';
import { useApp } from '../../context/AppContext';

interface FileUploadProps {
  onFileSelected: (file: File) => void;
  onOntologySelected: (file: File) => void;
}

export function FileUpload({ onFileSelected, onOntologySelected }: FileUploadProps) {
  const { dispatch } = useApp();
  const fileRef = useRef<HTMLInputElement>(null);
  const ontologyRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState('');
  const [ontologyName, setOntologyName] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    onFileSelected(file);
  };

  const handleOntologyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith('.json') && !file.name.endsWith('.owl')) {
      showError(dispatch, 'Only JSON and OWL ontology files are supported');
      e.target.value = '';
      setOntologyName('');
      return;
    }
    setOntologyName(file.name);
    onOntologySelected(file);
  };

  return (
    <>
      <input ref={fileRef} type="file" accept=".pdf,.txt,.json,.csv" className="hidden" onChange={handleFileChange} />
      <input ref={ontologyRef} type="file" accept=".json,.owl" className="hidden" onChange={handleOntologyChange} />
      <div className="flex gap-2">
        <button className="btn btn-primary btn-sm flex-1" onClick={() => fileRef.current?.click()}>Select File</button>
        <button className="btn btn-ghost btn-sm" onClick={() => ontologyRef.current?.click()}>Ontology</button>
      </div>
      {fileName && <div className="text-xs mt-1 opacity-70">Selected: {fileName}</div>}
      {ontologyName && <div className="text-xs mt-1 opacity-70">Ontology: {ontologyName}</div>}
    </>
  );
}
