import { useState } from 'react';
import axios from 'axios';

export default function ImageGenerator() {
  const [prompt, setPrompt] = useState('');
  const [steps, setSteps] = useState(25);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const generateImage = async () => {
    setLoading(true);
    setImage(null);
    try {
      const response = await axios.post('http://localhost:8000/generate', {
        prompt,
        steps,
        guidance_scale: guidanceScale,
      }, { responseType: 'blob' }); // 이미지 파일 받기

      const imageUrl = URL.createObjectURL(response.data);
      setImage(imageUrl);
    } catch (error) {
      console.error('이미지 생성 오류:', error);
    }
    setLoading(false);
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold">한국어로 이미지생성 (한복 LoRA)</h1>
      <input
        type="text"
        placeholder="프롬프트 입력"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        className="border p-2 my-2 w-full"
      />
      <div className="my-2">
        <label>Steps: {steps}</label>
        <input
          type="range"
          min="10"
          max="50"
          value={steps}
          onChange={(e) => setSteps(e.target.value)}
        />
      </div>
      <div className="my-2">
        <label>Guidance Scale: {guidanceScale}</label>
        <input
          type="range"
          min="1"
          max="20"
          step="0.5"
          value={guidanceScale}
          onChange={(e) => setGuidanceScale(e.target.value)}
        />
      </div>
      <button
        onClick={generateImage}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        {loading ? '이미지 생성 중...' : '이미지 생성'}
      </button>
      {image && (
        <div className="my-4">
          <h2 className="text-xl font-bold">결과 이미지</h2>
          <img src={image} alt="생성된 이미지" className="border" />
        </div>
      )}
    </div>
  );
}