import { useState } from "react";
import Plot from "react-plotly.js";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { Upload, Brain, BarChart3, Download } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const API_BASE_URL = "http://localhost:8000";

const Dashboard = () => {
  const [file, setFile] = useState<File | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  const [edaResults, setEdaResults] = useState<any>(null);
  const [preprocessReport, setPreprocessReport] = useState<any>(null);
  const [trainingResults, setTrainingResults] = useState<any>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const uploadDataset = async () => {
    if (!file) {
      toast({ title: "Error", description: "Please select a CSV file", variant: "destructive" });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload_dataset`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to upload dataset");
      }
      
      const data = await response.json();
      setDatasetInfo(data.info);
      toast({ title: "Success", description: "Dataset uploaded successfully!" });
    } catch (error: any) {
      console.error("Upload error:", error);
      toast({ 
        title: "Error", 
        description: error.message || "Failed to upload dataset. Make sure the backend is running on http://localhost:8000", 
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
    }
  };

  const runEDA = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/run_eda`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to run EDA");
      }

      const data = await response.json();
      const normalizedResults = data?.results ?? data;
      setEdaResults(normalizedResults);
      toast({ title: "Success", description: "EDA completed!" });
    } catch (error: any) {
      console.error("EDA error:", error);
      toast({ title: "Error", description: error.message || "Failed to run EDA", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const parsePlotJSON = (plotJson?: string) => {
    if (!plotJson) return null;
    try {
      return JSON.parse(plotJson);
    } catch (error) {
      console.error("Failed to parse plot JSON", error);
      return null;
    }
  };

  const renderDataTable = (
    rows: Record<string, any>[] = [],
    columns: { label: string; accessor: string }[] = [],
    keyPrefix: string,
  ) => {
    if (!rows?.length || !columns.length) return null;

    const formatCell = (value: any) => {
      if (value === null || value === undefined || value === "") return "—";
      if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString() : value;
      return value;
    };

    return (
      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((col) => (
                <TableHead key={`${keyPrefix}-head-${col.accessor}`}>{col.label}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, rowIdx) => (
              <TableRow key={`${keyPrefix}-row-${rowIdx}`}>
                {columns.map((col) => (
                  <TableCell key={`${keyPrefix}-cell-${rowIdx}-${col.accessor}`}>
                    {formatCell(row[col.accessor])}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  };

  const preprocessData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/preprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetColumn || null }),
      });
      const data = await response.json();
      setPreprocessReport(data.report);
      toast({ title: "Success", description: "Preprocessing completed!" });
    } catch (error) {
      toast({ title: "Error", description: "Failed to preprocess data", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const trainModels = async () => {
    if (!targetColumn) {
      toast({ title: "Error", description: "Please specify target column", variant: "destructive" });
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/train_all_models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetColumn }),
      });
      const data = await response.json();
      setTrainingResults(data);
      toast({ title: "Success", description: "Model training completed!" });
    } catch (error) {
      toast({ title: "Error", description: "Failed to train models", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold">ML Analytics Dashboard</h1>
          <p className="text-muted-foreground">Upload your dataset and train machine learning models</p>
        </div>

        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="eda">EDA</TabsTrigger>
            <TabsTrigger value="preprocess">Preprocess</TabsTrigger>
            <TabsTrigger value="train">Train Models</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload Dataset
                </CardTitle>
                <CardDescription>Upload your CSV file to begin analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="file">Choose CSV File</Label>
                  <Input id="file" type="file" accept=".csv" onChange={handleFileChange} />
                </div>
                <Button onClick={uploadDataset} disabled={loading || !file} className="w-full">
                  {loading ? "Uploading..." : "Upload Dataset"}
                </Button>

                {datasetInfo && (
                  <div className="mt-4 space-y-4 rounded-lg border p-4">
                    <div className="grid gap-4 sm:grid-cols-3">
                      <div>
                        <p className="text-sm text-muted-foreground">Rows</p>
                        <p className="text-2xl font-semibold">{datasetInfo.rows?.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Columns</p>
                        <p className="text-2xl font-semibold">{datasetInfo.columns}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">File</p>
                        <p className="font-medium break-words">{datasetInfo.filename}</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-semibold">Active Columns</p>
                      <div className="flex flex-wrap gap-2">
                        {datasetInfo.column_names?.map((col: string) => (
                          <Badge key={col} variant="secondary">
                            {col}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {datasetInfo.cleanup_summary && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-semibold">Automatic Cleanup</p>
                          <Badge variant={datasetInfo.cleanup_summary.columns_removed ? "destructive" : "outline"}>
                            {datasetInfo.cleanup_summary.columns_removed ?? 0} removed
                          </Badge>
                        </div>
                        {datasetInfo.cleanup_summary.columns_removed ? (
                          <div className="space-y-2">
                            <p className="text-sm text-muted-foreground">Removed columns (ID / index / constant)</p>
                            <div className="flex flex-wrap gap-2">
                              {datasetInfo.cleanup_summary.removed_columns?.map((col: any) => (
                                <Badge key={col.name} variant="outline">
                                  {col.name}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm text-muted-foreground">Dataset looks clean. No ID/index columns were removed.</p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="eda" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Exploratory Data Analysis
                </CardTitle>
                <CardDescription>Analyze your dataset statistics and visualizations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={runEDA} disabled={loading || !datasetInfo} className="w-full">
                  {loading ? "Running EDA..." : "Run EDA"}
                </Button>

                {edaResults && (
                  <div className="space-y-6">
                    {edaResults.status_messages && (
                      <Card>
                        <CardHeader>
                          <CardTitle>Status Updates</CardTitle>
                          <CardDescription>Live checkpoints from the backend pipeline.</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                            {edaResults.status_messages.map((status: string, idx: number) => (
                              <li key={`${status}-${idx}`}>{status}</li>
                            ))}
                          </ul>
                        </CardContent>
                      </Card>
                    )}

                    <div className="grid gap-4 lg:grid-cols-2">
                      {edaResults.structural_summary_rows?.length ? (
                        <Card className="h-full">
                          <CardHeader>
                            <CardTitle>Structural Summary</CardTitle>
                            <CardDescription>Key metrics describing the dataset footprint.</CardDescription>
                          </CardHeader>
                          <CardContent>
                            {renderDataTable(
                              edaResults.structural_summary_rows,
                              [
                                { label: "Metric", accessor: "metric" },
                                { label: "Value", accessor: "value" },
                              ],
                              "structural",
                            )}
                          </CardContent>
                        </Card>
                      ) : null}

                      {edaResults.missing_values_rows?.length ? (
                        <Card className="h-full">
                          <CardHeader>
                            <CardTitle>Missing Values</CardTitle>
                            <CardDescription>Full-column inventory with percentages.</CardDescription>
                          </CardHeader>
                          <CardContent>
                            {renderDataTable(
                              edaResults.missing_values_rows,
                              [
                                { label: "Column", accessor: "Column Name" },
                                { label: "Missing Count", accessor: "Missing Count" },
                                { label: "Missing %", accessor: "Missing Percentage" },
                              ],
                              "missing-values",
                            )}
                          </CardContent>
                        </Card>
                      ) : null}
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2">
                      {edaResults.basic_statistics_rows?.length ? (
                        <Card className="h-full">
                          <CardHeader>
                            <CardTitle>Basic Statistics</CardTitle>
                            <CardDescription>Distribution metrics for every numeric field.</CardDescription>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            {renderDataTable(
                              edaResults.basic_statistics_rows,
                              (edaResults.basic_statistics_rows[0]
                                ? Object.keys(edaResults.basic_statistics_rows[0]).map((key) => ({
                                    label: key,
                                    accessor: key,
                                  }))
                                : []),
                              "basic-stats",
                            )}
                          </CardContent>
                        </Card>
                      ) : null}

                      <Card className="h-full">
                        <CardHeader>
                          <CardTitle>Cleaning Report</CardTitle>
                          <CardDescription>Automatic imputation and column cleanup overview.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          {edaResults.cleaning_report_details?.dropped_rows_count ? (
                            <Badge variant="destructive">
                              Dropped Rows: {edaResults.cleaning_report_details.dropped_rows_count}
                            </Badge>
                          ) : (
                            <p className="text-sm text-muted-foreground">No rows were dropped during cleaning.</p>
                          )}

                          {edaResults.cleaning_report_details?.dropped_columns?.length ? (
                            <div className="space-y-2">
                              <p className="text-sm font-medium">Columns Removed</p>
                              {renderDataTable(
                                edaResults.cleaning_report_details.dropped_columns,
                                [
                                  { label: "Column", accessor: "column" },
                                  { label: "Reason", accessor: "reason" },
                                ],
                                "dropped-columns",
                              )}
                            </div>
                          ) : (
                            <p className="text-sm text-muted-foreground">No columns were removed in this pass.</p>
                          )}

                          {edaResults.cleaning_report_details?.imputed_columns?.length ? (
                            <div className="space-y-2">
                              <p className="text-sm font-medium">Imputed Columns</p>
                              {renderDataTable(
                                edaResults.cleaning_report_details.imputed_columns,
                                [
                                  { label: "Column", accessor: "column" },
                                  { label: "Strategy", accessor: "strategy" },
                                  { label: "NaN Ratio", accessor: "nan_ratio" },
                                ],
                                "imputed-columns",
                              )}
                            </div>
                          ) : null}
                        </CardContent>
                      </Card>
                    </div>

                    {edaResults.plots?.length > 0 && (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <div>
                            <h3 className="font-semibold">Visualizations</h3>
                            <p className="text-sm text-muted-foreground">
                              {edaResults.total_plots ?? edaResults.plots.length} interactive Plotly graphs
                            </p>
                          </div>
                          <Badge variant="outline">Correlation ≥ {edaResults.correlation_threshold ?? 0.8}</Badge>
                        </div>
                        <div className="grid gap-6 lg:grid-cols-2">
                          {edaResults.plots.map((plot: any, idx: number) => {
                            const parsed = parsePlotJSON(plot.data);
                            if (!parsed) return null;
                            const layout = {
                              ...parsed.layout,
                              title: plot.title || parsed.layout?.title,
                              autosize: true,
                              height: 380,
                              margin: {
                                t: 60,
                                r: 24,
                                b: 60,
                                l: 60,
                                ...(parsed.layout?.margin || {}),
                              },
                            };
                            const config = { responsive: true, displaylogo: false, ...(parsed.config || {}) };

                            return (
                              <Card className="h-full" key={`${plot.title ?? "plot"}-${idx}`}>
                                <CardHeader>
                                  <CardTitle className="text-lg">
                                    {plot.title || parsed.layout?.title?.text || `Plot ${idx + 1}`}
                                  </CardTitle>
                                  {plot.correlation && (
                                    <CardDescription>Correlation: {plot.correlation}</CardDescription>
                                  )}
                                </CardHeader>
                                <CardContent className="h-[420px]">
                                  <Plot data={parsed.data} layout={layout} config={config} style={{ width: "100%", height: "100%" }} />
                                </CardContent>
                              </Card>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="preprocess" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Preprocess Data</CardTitle>
                <CardDescription>Clean and prepare your dataset for training</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="target">Target Column (optional for preprocessing)</Label>
                  <Input
                    id="target"
                    placeholder="e.g., target, label, price"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                  />
                </div>
                <Button onClick={preprocessData} disabled={loading || !datasetInfo} className="w-full">
                  {loading ? "Preprocessing..." : "Preprocess Dataset"}
                </Button>

                {preprocessReport && (
                  <div className="space-y-4">
                    <Card>
                      <CardHeader>
                        <CardTitle>Step 1 • Missing Values</CardTitle>
                        <CardDescription>Numerical columns → mean, categorical columns → mode.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground">
                          {preprocessReport.missing_values_handled
                            ? "All missing values have been imputed."
                            : "No missing value handling was required."}
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Step 2 • Outlier Treatment</CardTitle>
                        <CardDescription>Columns are capped via Winsorization or IQR.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        {Object.keys(preprocessReport.outlier_report ?? {}).length ? (
                          renderDataTable(
                            Object.entries(preprocessReport.outlier_report).map(([column, method]) => ({
                              column,
                              method,
                            })),
                            [
                              { label: "Column", accessor: "column" },
                              { label: "Method", accessor: "method" },
                            ],
                            "outlier-report",
                          )
                        ) : (
                          <p className="text-sm text-muted-foreground">No outliers required adjustment.</p>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Step 3 • Encoding</CardTitle>
                        <CardDescription>Label Encoding applied to categorical features.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        {Object.keys(preprocessReport.encoding_report ?? {}).length ? (
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(preprocessReport.encoding_report).map(([column, method]) => (
                              <Badge key={column} variant="secondary">
                                {column}: {method}
                              </Badge>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-muted-foreground">No categorical columns detected.</p>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Step 4 • Scaling</CardTitle>
                        <CardDescription>StandardScaler applied to numeric fields (excluding target).</CardDescription>
                      </CardHeader>
                      <CardContent>
                        {preprocessReport.preprocessing_summary_table?.some((row: any) => row.action === "scaled") ? (
                          <div className="flex flex-wrap gap-2">
                            {preprocessReport.preprocessing_summary_table
                              .filter((row: any) => row.action === "scaled")
                              .map((row: any) => (
                                <Badge key={`scaled-${row.column}`} variant="outline">
                                  {row.column}
                                </Badge>
                              ))}
                          </div>
                        ) : (
                          <p className="text-sm text-muted-foreground">No numerical features were scaled.</p>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Step 5 • Summary Table</CardTitle>
                        <CardDescription>Complete log of preprocessing actions.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {preprocessReport.preprocessing_summary_table?.length ? (
                          renderDataTable(
                            preprocessReport.preprocessing_summary_table,
                            [
                              { label: "Column", accessor: "column" },
                              { label: "Type", accessor: "original_type" },
                              { label: "Action", accessor: "action" },
                              { label: "Method", accessor: "method" },
                            ],
                            "preprocess-summary",
                          )
                        ) : (
                          <p className="text-sm text-muted-foreground">No preprocessing actions recorded.</p>
                        )}

                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Final shape</span>
                          <Badge variant="secondary">
                            {preprocessReport.final_shape?.rows} rows × {preprocessReport.final_shape?.columns} columns
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="train" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Train Models
                </CardTitle>
                <CardDescription>Train multiple ML models and compare results</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="target-train">Target Column (required)</Label>
                  <Input
                    id="target-train"
                    placeholder="e.g., target, label, price"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                  />
                </div>
                <Button onClick={trainModels} disabled={loading || !datasetInfo || !targetColumn} className="w-full">
                  {loading ? "Training Models..." : "Train All Models"}
                </Button>

                {trainingResults && (
                  <div className="space-y-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <h3 className="font-semibold mb-2">Problem Type: {trainingResults.problem_type}</h3>
                    </div>

                    <div className="grid gap-4">
                      {trainingResults.results?.map((result: any, idx: number) => (
                        <Card key={idx}>
                          <CardHeader>
                            <CardTitle>{result.model}</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-2">
                              {result.accuracy && <p>Accuracy: {result.accuracy.toFixed(4)}</p>}
                              {result.f1_score && <p>F1 Score: {result.f1_score.toFixed(4)}</p>}
                              {result.r2_score && <p>R² Score: {result.r2_score.toFixed(4)}</p>}
                              {result.rmse && <p>RMSE: {result.rmse.toFixed(4)}</p>}
                              
                              {result.confusion_matrix_url && (
                                <img src={`${API_BASE_URL}${result.confusion_matrix_url}`} alt="Confusion Matrix" className="w-full rounded-lg mt-2" />
                              )}
                              
                              {result.model_path && (
                                <Button variant="outline" size="sm" className="mt-2" asChild>
                                  <a href={`${API_BASE_URL}/download_model/${result.model_path.split('/').pop()}`} download>
                                    <Download className="h-4 w-4 mr-2" />
                                    Download Model
                                  </a>
                                </Button>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Dashboard;
