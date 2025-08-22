import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '../../services/api';

interface KPIs {
  totalRevenue: number;
  totalCLV: number;
  avgCLV: number;
  totalProfit: number;
  avgProfitMargin: number;
  retentionRate: number;
  churnRate: number;
  activeCustomers: number;
  totalCustomers: number;
}

interface GrowthTrend {
  month: string;
  revenue: number;
  revenueGrowthRate: number;
}

interface CustomerDistribution {
  byCLVTier: Record<string, number>;
  bySegment: Record<string, number>;
}

interface DashboardState {
  kpis: KPIs | null;
  growthTrend: GrowthTrend[];
  customerDistribution: CustomerDistribution | null;
  loading: boolean;
  error: string | null;
}

const initialState: DashboardState = {
  kpis: null,
  growthTrend: [],
  customerDistribution: null,
  loading: false,
  error: null,
};

export const fetchDashboardData = createAsyncThunk(
  'dashboard/fetchData',
  async () => {
    const response = await api.getDashboardData();
    return response;
  }
);

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    resetDashboard: (state) => {
      state.kpis = null;
      state.growthTrend = [];
      state.customerDistribution = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDashboardData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDashboardData.fulfilled, (state, action) => {
        state.loading = false;
        state.kpis = action.payload.kpis;
        state.growthTrend = action.payload.growthTrend;
        state.customerDistribution = action.payload.customerDistribution;
      })
      .addCase(fetchDashboardData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch dashboard data';
      });
  },
});

export const { resetDashboard } = dashboardSlice.actions;
export default dashboardSlice.reducer;