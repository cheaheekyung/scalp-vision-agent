import React, { useState, useEffect, useRef } from 'react';

import {
  Users,
  Calendar,
  Activity,
  Search,
  Plus,
  ChevronRight,
  ArrowLeft,
  FileText,
  Microscope,
  Save,
} from 'lucide-react';

// ------------------------------
// 공통 설정/헬퍼
// ------------------------------
const API_BASE = 'http://localhost:8000';

const apiCall = async (endpoint, method = 'GET', body = null) => {
  try {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body) options.body = JSON.stringify(body);

    const response = await fetch(`${API_BASE}${endpoint}`, options);
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn(`API 호출 실패 (${endpoint})`, error);
    return null;
  }
};

const Loading = () => (
  <div className="flex justify-center items-center h-40">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
  </div>
);

const calculateAgeFromBirth = (birthDateStr) => {
  if (!birthDateStr) return null;
  try {
    const b = new Date(birthDateStr);
    const today = new Date();
    let age = today.getFullYear() - b.getFullYear();
    const m = today.getMonth() - b.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < b.getDate())) {
      age -= 1;
    }
    return age >= 0 ? age : null;
  } catch {
    return null;
  }
};

// 백엔드 User.gender → profile.gender("M" | "F" | "U") 정규화
const normalizeGenderForProfile = (gender) => {
  if (!gender) return 'U';
  if (gender === 'M' || gender === 'F' || gender === 'U') return gender;
  if (gender === 'male') return 'M';
  if (gender === 'female') return 'F';
  return 'U';
};

// ------------------------------
// 메인 App 컴포넌트
// ------------------------------
export default function App() {
  const [currentView, setCurrentView] = useState('userList'); // userList | userDetail | analysis
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedVisit, setSelectedVisit] = useState(null); // /visits/{id}/full 응답
  const [users, setUsers] = useState([]);
  const [visits, setVisits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [analysisCache, setAnalysisCache] = useState({});

  // 초기 유저 로딩
  useEffect(() => {
    fetchUsers();
  }, []);

  // ------------------------------
  // 유저 목록 / 검색
  // ------------------------------
  const fetchUsers = async (query = '') => {
    setLoading(true);
    const endpoint = query
      ? `/users/search?name=${encodeURIComponent(query)}`
      : '/users';

    const data = await apiCall(endpoint, 'GET');
    if (Array.isArray(data)) {
      setUsers(data);
    } else {
      setUsers([]);
    }
    setLoading(false);
  };

  const handleSearch = (e) => {
    e.preventDefault();
    fetchUsers(searchTerm);
  };

  // 신규 유저 생성 (아주 간단한 prompt 방식)
  const handleCreateUser = async () => {
    const name = window.prompt('고객 이름을 입력하세요');
    if (!name) return;

    let gender = window.prompt('성별을 입력하세요 (M/F/U)', 'U');
    gender = (gender || 'U').toUpperCase();
    if (!['M', 'F', 'U'].includes(gender)) {
      gender = 'U';
    }

    const birth = window.prompt('생년월일 (YYYY-MM-DD, 선택)', '');

    const payload = {
      name,
      gender,
      birth_date: birth || null,
    };

    const user = await apiCall('/users', 'POST', payload);
    if (user) {
      // 목록 맨 앞에 추가
      setUsers((prev) => [user, ...prev]);
      alert(`고객 생성 완료 (#${user.user_id} ${user.name})`);
    } else {
      alert('고객 생성 실패');
    }
  };

  // ------------------------------
  // 유저 선택 / 방문 목록
  // ------------------------------
  const handleUserClick = async (user) => {
    setLoading(true);
    setSelectedUser(user);

    const data = await apiCall(`/users/${user.user_id}/visits`, 'GET');
    if (Array.isArray(data)) {
      setVisits(data);
    } else {
      setVisits([]);
    }

    setCurrentView('userDetail');
    setLoading(false);
  };

  // 선택된 유저에 새 방문 생성
  const handleCreateVisit = async () => {
    if (!selectedUser) {
      alert('먼저 고객을 선택하세요.');
      return;
    }

    const dateStr =
      window.prompt(
        '방문 날짜를 입력하세요 (YYYY-MM-DD)',
        new Date().toISOString().slice(0, 10),
      ) || new Date().toISOString().slice(0, 10);

    const note = window.prompt('방문 메모 (선택)', '') || '';

    const payload = {
      user_id: selectedUser.user_id,
      visit_date: dateStr,
      note,
    };

    const visit = await apiCall('/visits', 'POST', payload);
    if (visit) {
      setVisits((prev) => [visit, ...prev]);
      alert(`Visit 생성 완료 (#${visit.visit_id})`);
    } else {
      alert('Visit 생성 실패');
    }
  };

  // ------------------------------
  // 방문 클릭 → /visits/{id}/full
  // ------------------------------
  const handleVisitClick = async (visit) => {
    setLoading(true);
    const data = await apiCall(`/visits/${visit.visit_id}/full`, 'GET');
    if (data && data.visit) {
      setSelectedVisit(data); // { user, visit, report }
      setCurrentView('analysis');
    } else {
      alert('방문 상세 정보를 가져오지 못했습니다.');
    }
    setLoading(false);
  };

  // ------------------------------
  // View 1: 유저 목록
  // ------------------------------
  const renderUserList = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">고객 관리</h2>
        <button
          onClick={handleCreateUser}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
        >
          <Plus size={18} /> 신규 고객 등록
        </button>
      </div>

      {/* 검색 바 */}
      <form onSubmit={handleSearch} className="relative">
        <input
          type="text"
          placeholder="고객 이름 검색..."
          className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none shadow-sm"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <Search className="absolute.left-3 top-3.5 text-gray-400" size={20} />
      </form>

      {/* 유저 테이블 */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        {loading ? (
          <Loading />
        ) : (
          <table className="w-full text-left">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="p-4 text-sm font-medium text-gray-500">ID</th>
                <th className="p-4 text-sm font-medium text-gray-500">이름</th>
                <th className="p-4 text-sm font-medium text-gray-500">성별</th>
                <th className="p-4 text-sm font-medium text-gray-500">
                  생년월일
                </th>
                <th className="p-4 text-sm font-medium text-gray-500">
                  등록일
                </th>
                <th className="p-4 text-sm font-medium text-gray-500" />
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr
                  key={user.user_id}
                  onClick={() => handleUserClick(user)}
                  className="hover:bg-blue-50 cursor-pointer transition border-b border-gray-50 last:border-0"
                >
                  <td className="p-4 text-gray-600">#{user.user_id}</td>
                  <td className="p-4 font-bold text-gray-800">{user.name}</td>
                  <td className="p-4 text-gray-600">
                    {user.gender === 'M'
                      ? '남성'
                      : user.gender === 'F'
                      ? '여성'
                      : '미상'}
                  </td>
                  <td className="p-4 text-gray-600">{user.birth_date || '-'}</td>
                  <td className="p-4 text-gray-500 text-sm">
                    {user.created_at
                      ? new Date(user.created_at).toLocaleDateString()
                      : '-'}
                  </td>
                  <td className="p-4 text-right">
                    <ChevronRight size={18} className="text-gray-400" />
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr>
                  <td colSpan={6} className="p-8 text-center text-gray-500">
                    검색 결과가 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );

  // ------------------------------
  // View 2: 유저 상세 (방문 목록)
  // ------------------------------
  const renderUserDetail = () => (
    <div className="space-y-6">
      <button
        onClick={() => setCurrentView('userList')}
        className="flex items-center text-gray-500 hover:text-blue-600 transition"
      >
        <ArrowLeft size={18} className="mr-1" /> 고객 목록으로 돌아가기
      </button>

      {/* 프로필 카드 */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-2xl font-bold">
            {selectedUser.name[0]}
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              {selectedUser.name}
            </h2>
            <p className="text-gray-500">
              {selectedUser.gender === 'M'
                ? '남성'
                : selectedUser.gender === 'F'
                ? '여성'
                : '미상'}{' '}
              | {selectedUser.birth_date || '생년월일 정보 없음'}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-500">고객 ID</p>
          <p className="font-mono font-medium text-gray-800">
            #{selectedUser.user_id}
          </p>
        </div>
      </div>

      {/* 방문 기록 섹션 */}
      <div className="flex justify-between items-center mt-8">
        <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <Calendar size={20} /> 방문 이력
        </h3>
        <button
          onClick={handleCreateVisit}
          className="text-sm bg-gray-800 text-white px-3 py-1.5 rounded-lg hover:bg-gray-900 transition"
        >
          + 새 방문 기록
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {visits.map((visit) => (
          <div
            key={visit.visit_id}
            onClick={() => handleVisitClick(visit)}
            className="bg-white p-5 rounded-xl border border-gray-200 hover:border-blue-400 hover:shadow-md cursor-pointer transition group"
          >
            <div className="flex justify-between items-start mb-3">
              <span className="bg-blue-50 text-blue-700 text-xs font-bold px-2 py-1 rounded">
                Visit #{visit.visit_id}
              </span>
              <ChevronRight
                size={18}
                className="text-gray-300 group-hover:text-blue-500"
              />
            </div>
            <p className="text-lg font-bold text-gray-800 mb-1">
              {visit.visit_date}
            </p>
            <p className="text-gray-500 text-sm line-clamp-2">
              {visit.note || '메모 없음'}
            </p>
          </div>
        ))}
        {visits.length === 0 && (
          <div className="col-span-full py-10 text-center text-gray-400 bg-gray-50 rounded-xl border border-dashed border-gray-300">
            방문 기록이 없습니다.
          </div>
        )}
      </div>
    </div>
  );

  // ------------------------------
  // View 3: 분석 / 리포트
  // ------------------------------
  const AnalysisView = ({ fullData, analysisCache, onSaveAnalysis }) => {
    const { user, visit, report } = fullData;

    const [form, setForm] = useState({
      sample_id: '',
      location: 'TH',
      value_1: 0,
      value_2: 0,
      value_3: 0,
      value_4: 0,
      value_5: 0,
      value_6: 0,
    });

    const cached = analysisCache?.[visit.visit_id];
    const [localReport, setLocalReport] = useState(cached || report || null);
    const [analyzing, setAnalyzing] = useState(false);

    const [uploading, setUploading] = useState(false);
    const fileInputRef = useRef(null);

    const handleSliderChange = (key, value) => {
      setForm((prev) => ({ ...prev, [key]: parseInt(value, 10) }));
    };

    const runAnalysis = async () => {
      setAnalyzing(true);

      const age = calculateAgeFromBirth(user.birth_date);

      const condition = {
        sample_id: form.sample_id || `visit_${visit.visit_id}_manual`,
        location: form.location,
        value_1: form.value_1,
        value_2: form.value_2,
        value_3: form.value_3,
        value_4: form.value_4,
        value_5: form.value_5,
        value_6: form.value_6,
      };

      const profile = {
        gender: normalizeGenderForProfile(user.gender),
        age: age,
        shampoo_frequency: null,
        perm_frequency: null,
        dye_frequency: null,
      };

      const payload = { condition, profile };

      const result = await apiCall(
        `/visits/${visit.visit_id}/analyze-demo`,
        'POST',
        payload,
      );

      if (result) {
        // 분석 후 저장된 VisitReport 다시 가져오기
        const reportData = await apiCall(
          `/visits/${visit.visit_id}/report`,
          'GET',
        );

        const nextReport = {
          risk_score: result.risk_score,
          risk_level: result.risk_level,
          summary: result.summary,
          details: result.details,
          recommendations: result.recommendations || [],
          history_message: result.history_message || null,
          plan_text: result.plan_text || null,
          report_text:
            (reportData && reportData.report_text) ||
            result.details ||
            result.summary ||
            '리포트 텍스트를 불러오지 못했습니다.',
        };

        setLocalReport(nextReport);

        if (onSaveAnalysis) {
          onSaveAnalysis(visit.visit_id, nextReport);
        }
      } else {
        alert('분석 요청 실패');
      }

      setAnalyzing(false);
    };

    const handleImageButtonClick = () => {
      if (!fileInputRef.current) return;
      fileInputRef.current.click();
    };

    const handleImageChange = async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;

      setUploading(true);

      try {
        const age = calculateAgeFromBirth(user.birth_date);
        const gender = normalizeGenderForProfile(user.gender); // "M" | "F" | "U"

        const params = new URLSearchParams();
        if (gender) params.append('gender', gender);
        if (age !== null && age !== undefined) {
          params.append('age', String(age));
        }

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(
          `${API_BASE}/visits/${visit.visit_id}/analyze-image?${params.toString()}`,
          {
            method: 'POST',
            body: formData,
          },
        );

        if (!response.ok) {
          throw new Error(`이미지 분석 실패: ${response.status}`);
        }

        const result = await response.json();

        const reportData = await apiCall(
          `/visits/${visit.visit_id}/report`,
          'GET',
        );

        const nextReport = {
          risk_score: result.risk_score,
          risk_level: result.risk_level,
          summary: result.summary,
          details: result.details,
          recommendations: result.recommendations || [],
          history_message: result.history_message || null,
          plan_text: result.plan_text || null,
          report_text:
            (reportData && reportData.report_text) ||
            result.details ||
            result.summary ||
            '리포트 텍스트를 불러오지 못했습니다.',
        };

        setLocalReport(nextReport);

        if (onSaveAnalysis) {
          onSaveAnalysis(visit.visit_id, nextReport);
        }
      } catch (err) {
        console.error(err);
        alert('이미지 기반 분석 중 오류가 발생했습니다.');
      } finally {
        setUploading(false);
        if (event.target) {
          event.target.value = '';
        }
      }
    };

    return (
      <div className="space-y-6">
        <button
          onClick={() => setCurrentView('userDetail')}
          className="flex items-center text-gray-500 hover:text-blue-600 transition"
        >
          <ArrowLeft size={18} className="mr-1" /> 방문 목록으로 돌아가기
        </button>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
          {/* 왼쪽: 이미지 분석 + 수동 입력 */}
          <div className="space-y-6">
            {/* 1) 이미지 분석 메인 카드 */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
              <h3 className="text-lg font-bold text-gray-800 mb-2 flex items-center gap-2">
                <Microscope className="text-blue-600" /> 이미지 기반 분석
              </h3>
              <p className="text-sm text-gray-500 mb-4">
                두피 스코프 이미지를 업로드하면 CNN + Agent가 자동으로 상태를
                분석하고 리포트를 생성합니다.
              </p>

              <div
                onClick={handleImageButtonClick}
                className={`mt-2 border-2 border-dashed rounded-xl px-4 py-10 text-center cursor-pointer transition
                  ${
                    uploading
                      ? 'border-blue-300 bg-blue-50'
                      : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50/40'
                  }`}
              >
                <Microscope
                  size={32}
                  className={`mx-auto mb-3 ${
                    uploading ? 'text-blue-500' : 'text-gray-400'
                  }`}
                />
                <p className="font-semibold text-gray-700 mb-1">
                  {uploading ? '이미지 분석 중...' : '이미지 파일을 선택하세요'}
                </p>
                <p className="text-xs text-gray-500">
                  클릭해서 파일 선택 (드래그 앤 드롭 UI는 추후 추가 예정)
                </p>
              </div>

              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleImageChange}
                className="hidden"
              />
            </div>

            {/* 2) 수동 입력 (옵션) */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
              <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Activity className="text-gray-500" /> 수동 입력 (선택)
              </h3>

              <p className="text-xs text-gray-400 mb-4">
                이미지가 없는 경우나 테스트용으로 value_1~6 값을 직접 입력해
                규칙 기반 분석을 실행할 수 있습니다.
              </p>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  샘플 ID (선택)
                </label>
                <input
                  type="text"
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                  placeholder="미입력 시 visit 기반 자동 생성"
                  value={form.sample_id}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      sample_id: e.target.value,
                    }))
                  }
                />
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  촬영 위치
                </label>
                <select
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                  value={form.location}
                  onChange={(e) =>
                    setForm((prev) => ({ ...prev, location: e.target.value }))
                  }
                >
                  <option value="TH">정수리(TH)</option>
                  <option value="LH">좌측(LH)</option>
                  <option value="RH">우측(RH)</option>
                  <option value="BH">후두부(BH)</option>
                </select>
              </div>

              {/* value_1~6 슬라이더 */}
              <div className="space-y-4">
                {[
                  { key: 'value_1', label: 'value_1 (각질)' },
                  { key: 'value_2', label: 'value_2 (피지)' },
                  { key: 'value_3', label: 'value_3 (모낭 사이 홍반)' },
                  { key: 'value_4', label: 'value_4 (모낭 홍반/농포)' },
                  { key: 'value_5', label: 'value_5 (비듬)' },
                  { key: 'value_6', label: 'value_6 (탈모)' },
                ].map((item) => (
                  <div key={item.key}>
                    <div className="flex justify-between mb-1">
                      <label className="text-sm font-medium text-gray-700">
                        {item.label}
                      </label>
                      <span className="text-sm text-blue-600 font-bold">
                        {form[item.key]}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="3"
                      step="1"
                      value={form[item.key]}
                      onChange={(e) =>
                        handleSliderChange(item.key, e.target.value)
                      }
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400">
                      <span>정상 (0)</span>
                      <span>심함 (3)</span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6">
                <button
                  onClick={runAnalysis}
                  disabled={analyzing}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-xl transition flex justify-center items-center gap-2"
                >
                  {analyzing ? '수동 값으로 분석 중...' : '수동 값으로 AI 분석 실행'}
                </button>
              </div>
            </div>
          </div>

          {/* 오른쪽: 결과 리포트 */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 h-full">
              <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FileText className="text-green-600" /> 분석 리포트
              </h3>

              {!localReport ? (
                <div className="h-64 flex flex-col items-center justify-center text-gray-400">
                  <Activity size={48} className="mb-4 opacity-20" />
                  <p>아직 분석 리포트가 없습니다.</p>
                  <p className="text-sm">왼쪽에서 이미지 업로드 또는 수동 분석을 실행하세요.</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* 종합 위험도 카드 */}
                  <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-6 rounded-xl text-center border border-gray-200">
                    <p className="text-sm text-gray-500 mb-1">종합 위험도 점수</p>
                    <div className="text-5xl font-black text-gray-800 mb-2">
                          {localReport?.risk_score !== undefined && localReport?.risk_score !== null
                            ? localReport.risk_score.toFixed(1)   // 🔹 2.0, 2.3 이런 식으로 표시
                            : "-"}
                    </div>
                    <span
                      className={`inline-block px-3 py-1 rounded-full text-sm.font-bold
                        ${
                          localReport.risk_level === 'high'
                            ? 'bg-red-100 text-red-700'
                            : localReport.risk_level === 'medium'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-green-100 text-green-700'
                        }`}
                    >
                      {localReport.risk_level}
                    </span>
                  </div>

                  {/* 요약 섹션 (summary) */}
                  {localReport.summary && (
                    <div>
                      <h4 className="font-bold text-gray-800 mb-2">요약</h4>
                      <div className="bg-gray-50 p-4 rounded-lg text-gray-700 text-sm leading-relaxed whitespace-pre-wrap">
                        {localReport.summary}
                      </div>
                    </div>
                  )}

                  {/* 이전 방문과 비교 (history_message) */}
                  {localReport.history_message && (
                    <div>
                      <h4 className="font-bold text-gray-800 mb-2">
                        이전 방문과 비교
                      </h4>
                      <div className="bg-amber-50 p-4 rounded-lg text-gray-700 text-sm leading-relaxed whitespace-pre-wrap border border-amber-100">
                        {localReport.history_message}
                      </div>
                    </div>
                  )}

                  {/* 관리 플랜 (plan_text) */}
                  {localReport.plan_text && (
                    <div>
                      <h4 className="font-bold text-gray-800 mb-2">
                        관리 플랜 (1~3개월)
                      </h4>
                      <div className="bg-green-50 p-4 rounded-lg text-gray-700 text-sm.leading-relaxed whitespace-pre-wrap border border-green-100">
                        {localReport.plan_text}
                      </div>
                    </div>
                  )}

                  {/* 상세 리포트 (LLM 전체 텍스트) */}
                  <div>
                    <h4 className="font-bold text-gray-800 mb-2">상세 리포트</h4>
                    <div className="bg-blue-50 p-4 rounded-lg text-gray-700 text-sm.leading-relaxed whitespace-pre-wrap">
                      {localReport.report_text}
                    </div>
                  </div>

                  {/* 권장 사항 리스트 */}
                  {Array.isArray(localReport.recommendations) &&
                    localReport.recommendations.length > 0 && (
                      <div>
                        <h4 className="font-bold text-gray-800 mb-2">
                          권장 사항
                        </h4>
                        <ul className="space-y-2 text-sm">
                          {localReport.recommendations.map((rec, idx) => (
                            <li
                              key={idx}
                              className="bg-white border border-gray-100 rounded-lg p-3 shadow-sm"
                            >
                              <p className="font-semibold text-gray-800">
                                {rec.title}
                              </p>
                              <p className="text-gray-600 mt-1 text-sm">
                                {rec.description}
                              </p>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {/* 하단 버튼 */}
                  <div className="pt-4 border-t border-gray-100 flex justify-end">
                    <button className="text-gray-500 hover:text-gray-800 text-sm flex items-center gap-1">
                      <Save size={14} /> PDF 저장 (준비중)
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ------------------------------
  // 메인 레이아웃
  // ------------------------------
  return (
    <div className="flex h-screen bg-gray-50 font-sans text-gray-900">
      {/* 사이드바 */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col shadow-lg z-10">
        <div className="p-6 border-b border-gray-100">
          <div className="flex items-center gap-2 text-blue-600 font-black text-xl">
            <Activity />
            <span>ScalpVision</span>
          </div>
          <p className="text-xs text-gray-400 mt-1 ml-8">AI Agent Admin</p>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <button
            onClick={() => {
              setSelectedUser(null);
              setSelectedVisit(null);
              setCurrentView('userList');
            }}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition font-medium
              ${
                currentView === 'userList'
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
          >
            <Users size={20} /> 고객 관리
          </button>
          <div className="px-4 py-3 text-gray-400 flex items-center gap-3 cursor-not-allowed">
            <Calendar size={20} /> 예약 일정 (준비중)
          </div>
        </nav>

        <div className="p-4 border-t border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
              <Users size={16} className="text-gray-500" />
            </div>
            <div className="text-xs">
              <p className="font-bold text-gray-700">관리자 계정</p>
              <p className="text-gray-400">admin@scalp.ai</p>
            </div>
          </div>
        </div>
      </aside>

      {/* 컨텐츠 영역 */}
      <main className="flex-1 overflow-y-auto p-8">
        <div className="max-w-6xl mx-auto">
          {currentView === 'userList' && renderUserList()}
          {currentView === 'userDetail' && selectedUser && renderUserDetail()}
          {currentView === 'analysis' && selectedVisit && (
            <AnalysisView
              fullData={selectedVisit}
              analysisCache={analysisCache}
              onSaveAnalysis={(visitId, analysis) =>
                setAnalysisCache((prev) => ({
                  ...prev,
                  [visitId]: analysis,
                }))
              }
            />
          )}
        </div>
      </main>
    </div>
  );
}
