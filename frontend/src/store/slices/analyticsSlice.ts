import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { api } from '../../services/api';

interface CohortData {
  cohortMonth: string;
  retention: number[];
  revenue: number[];
  ltv: number;
}

interface ABTestResult {
  controlRate: number;
  treatmentRate: number;
  lift: number;
  pValue: number;
  isSignificant: boolean;
}

interface AnalyticsState {
  cohortData: CohortData[];
  abTestResults: ABTestResult | null;
  churnPredictions: any[];
  recommendations: any[];
  loading: boolean;
  error: string | null;
}

const initialState: AnalyticsState = {
  cohortData: [],
  abTestResults: null,
  churnPredictions: [],
  recommendations: [],
  loading: false,
  error: null,
};

export const fetchCohortAnalysis = createAsyncThunk(
  'analytics/fetchCohortAnalysis',
  async () => {
    const response = await api.getCohortAnalysis();
    return response;
  }
);

export const calculateABTest = createAsyncThunk(
  'analytics/calculateABTest',
  async (params: any) => {
    const response = await api.calculateABTest(params);
    return response;
  }
);

const analyticsSlice = createSlice({
  name: 'analytics',
  initialState,
  reducers: {
    clearAnalytics: (state) => {
      state.cohortData = [];
      state.abTestResults = null;
      state.churnPredictions = [];
      state.recommendations = [];
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchCohortAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCohortAnalysis.fulfilled, (state, action) => {
        state.loading = false;
        state.cohortData = action.payload;
      })
      .addCase(fetchCohortAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch cohort analysis';
      })
      .addCase(calculateABTest.fulfilled, (state, action) => {
        state.abTestResults = action.payload;
      });
  },
});

export const { clearAnalytics } = analyticsSlice.actions;
export default analyticsSlice.reducer;