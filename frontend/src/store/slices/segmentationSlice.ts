import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { api } from '../../services/api';

interface Segment {
  id: string;
  name: string;
  customerCount: number;
  avgRevenue: number;
  avgOrders: number;
  churnRate: number;
}

interface SegmentationState {
  segments: Segment[];
  selectedSegment: Segment | null;
  loading: boolean;
  error: string | null;
}

const initialState: SegmentationState = {
  segments: [],
  selectedSegment: null,
  loading: false,
  error: null,
};

export const fetchSegments = createAsyncThunk(
  'segmentation/fetchSegments',
  async (type: string) => {
    const response = await api.getSegments(type);
    return response;
  }
);

const segmentationSlice = createSlice({
  name: 'segmentation',
  initialState,
  reducers: {
    selectSegment: (state, action) => {
      state.selectedSegment = action.payload;
    },
    clearSegments: (state) => {
      state.segments = [];
      state.selectedSegment = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSegments.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSegments.fulfilled, (state, action) => {
        state.loading = false;
        state.segments = action.payload;
      })
      .addCase(fetchSegments.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch segments';
      });
  },
});

export const { selectSegment, clearSegments } = segmentationSlice.actions;
export default segmentationSlice.reducer;