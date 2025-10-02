import { useState, useRef } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);

  let text;
  if (!result && !selectedFile){
    text = "Drop your receipt/ PDF file here"
  }
  else if (!result && selectedFile){
    text = `Selected File: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(2)} KB)`
  }
  else{
    text = `The detected text is: ${result}`
  }
    
    
  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    const allowedFiles = droppedFiles.filter(file =>
      file.type === "application/pdf" || file.type.startsWith("image/")
    );

    if (allowedFiles.length === 0) {
      alert("Only PDF or image files are allowed!");
      return;
    }

    setSelectedFile(allowedFiles[0]);
  };


  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first!');
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setSelectedFile(null);
      setResult(data.texts.join(" "));
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleClear = () => {
      setSelectedFile(null);
      setResult(null);
  };

  const DragAndDrop = () => {
    return (
      <div className="flex flex-col items-center w-full">
        <div
          className="w-full h-80 bg-[rgb(146,138,121)] rounded-xl font-bold text-lg border-4 border-dashed border-gray-400 flex justify-center items-center p-20"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {text}
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="flex flex-col items-center min-h-screen bg-gradient-to-b from-blue-500 to-purple-600 text-white p-10">
        <h1 className="text-6xl font-bold mb-20">Receipt / PDF Reader</h1>
          <DragAndDrop />
        <div className="flex flex-row justify-center space-x-4 mt-6">
          <button
            className="bg-indigo-500 hover:bg-indigo-600 text-white font-bold mt-4 px-4 py-2 rounded"
            onClick={handleUpload}
          >
            Upload PDF
          </button>
          <button
            className="bg-indigo-500 hover:bg-indigo-600 text-white font-bold mt-4 px-6 py-2 rounded"
            onClick={handleClear}
          >
            Clear input
          </button>
        </div>
      </div>
    </>
  );
}

export default App;
