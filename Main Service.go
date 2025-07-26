package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/bigquery"
	"cloud.google.com/go/compute/apiv1"
	"cloud.google.com/go/monitoring/apiv3/v2"
	"github.com/gorilla/mux"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	computepb "google.golang.org/genproto/googleapis/cloud/compute/v1"
	monitoringpb "google.golang.org/genproto/googleapis/monitoring/v3"
)

// Resource represents a cloud resource
type Resource struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Type         string            `json:"type"`
	ProjectID    string            `json:"project_id"`
	Zone         string            `json:"zone"`
	Region       string            `json:"region"`
	Status       string            `json:"status"`
	CreatedAt    time.Time         `json:"created_at"`
	Labels       map[string]string `json:"labels"`
	MachineType  string            `json:"machine_type,omitempty"`
	DiskSizeGB   int64             `json:"disk_size_gb,omitempty"`
	VCPUs        int64             `json:"vcpus,omitempty"`
	MemoryMB     int64             `json:"memory_mb,omitempty"`
	CurrentCost  float64           `json:"current_cost"`
	Utilization  ResourceMetrics   `json:"utilization"`
}

// ResourceMetrics contains utilization metrics
type ResourceMetrics struct {
	CPUUtilization    float64 `json:"cpu_utilization"`
	MemoryUtilization float64 `json:"memory_utilization"`
	DiskUtilization   float64 `json:"disk_utilization"`
	NetworkBytes      int64   `json:"network_bytes"`
	LastUpdated       time.Time `json:"last_updated"`
}

// OptimizationRecommendation represents cost optimization suggestions
type OptimizationRecommendation struct {
	ResourceID      string  `json:"resource_id"`
	ResourceName    string  `json:"resource_name"`
	ResourceType    string  `json:"resource_type"`
	ProjectID       string  `json:"project_id"`
	CurrentCost     float64 `json:"current_cost_monthly"`
	RecommendedType string  `json:"recommended_type"`
	EstimatedCost   float64 `json:"estimated_cost_monthly"`
	MonthlySavings  float64 `json:"monthly_savings"`
	SavingsPercent  float64 `json:"savings_percent"`
	Reason          string  `json:"reason"`
	Confidence      string  `json:"confidence"`
	Impact          string  `json:"impact"`
	RiskLevel       string  `json:"risk_level"`
	AutoApplicable  bool    `json:"auto_applicable"`
}

// CostOptimizer main service struct
type CostOptimizer struct {
	bqClient         *bigquery.Client
	computeClient    *compute.InstancesClient
	monitoringClient *monitoring.MetricClient
	projectID        string
	datasetID        string
	tableID          string
}

// NewCostOptimizer creates a new cost optimizer service
func NewCostOptimizer(projectID, datasetID, tableID string) (*CostOptimizer, error) {
	ctx := context.Background()

	// Initialize BigQuery client
	bqClient, err := bigquery.NewClient(ctx, projectID)
	if err != nil {
		return nil, fmt.Errorf("failed to create BigQuery client: %v", err)
	}

	// Initialize Compute Engine client
	computeClient, err := compute.NewInstancesRESTClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create Compute client: %v", err)
	}

	// Initialize Cloud Monitoring client
	monitoringClient, err := monitoring.NewMetricClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create Monitoring client: %v", err)
	}

	return &CostOptimizer{
		bqClient:         bqClient,
		computeClient:    computeClient,
		monitoringClient: monitoringClient,
		projectID:        projectID,
		datasetID:        datasetID,
		tableID:          tableID,
	}, nil
}

// DiscoverResources scans GCP projects for compute resources
func (co *CostOptimizer) DiscoverResources() ([]Resource, error) {
	ctx := context.Background()
	var resources []Resource

	// Get list of zones
	zones, err := co.getProjectZones()
	if err != nil {
		return nil, fmt.Errorf("failed to get zones: %v", err)
	}

	// Discover instances in each zone
	for _, zone := range zones {
		req := &computepb.ListInstancesRequest{
			Project: co.projectID,
			Zone:    zone,
		}

		it := co.computeClient.List(ctx, req)
		for {
			instance, err := it.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				log.Printf("Error listing instances in zone %s: %v", zone, err)
				continue
			}

			// Get resource cost from BigQuery
			cost, err := co.getResourceCost(instance.GetName(), "Compute Engine")
			if err != nil {
				log.Printf("Error getting cost for instance %s: %v", instance.GetName(), err)
				cost = 0
			}

			// Get utilization metrics
			metrics, err := co.getResourceMetrics(zone, instance.GetName())
			if err != nil {
				log.Printf("Error getting metrics for instance %s: %v", instance.GetName(), err)
				metrics = ResourceMetrics{}
			}

			// Parse machine type to get specs
			machineType := instance.GetMachineType()
			machineTypeName := machineType[strings.LastIndex(machineType, "/")+1:]
			vcpus, memory := co.parseMachineType(machineTypeName)

			// Calculate total disk size
			var totalDiskGB int64
			for _, disk := range instance.GetDisks() {
				if disk.GetDiskSizeGb() > 0 {
					totalDiskGB += disk.GetDiskSizeGb()
				}
			}

			resource := Resource{
				ID:          fmt.Sprintf("%d", instance.GetId()),
				Name:        instance.GetName(),
				Type:        "compute-instance",
				ProjectID:   co.projectID,
				Zone:        zone,
				Region:      zone[:strings.LastIndex(zone, "-")],
				Status:      instance.GetStatus(),
				MachineType: machineTypeName,
				VCPUs:       vcpus,
				MemoryMB:    memory,
				DiskSizeGB:  totalDiskGB,
				CurrentCost: cost,
				Utilization: metrics,
				Labels:      instance.GetLabels(),
			}

			if instance.GetCreationTimestamp() != "" {
				if createdAt, err := time.Parse(time.RFC3339, instance.GetCreationTimestamp()); err == nil {
					resource.CreatedAt = createdAt
				}
			}

			resources = append(resources, resource)
		}
	}

	return resources, nil
}

// getProjectZones retrieves all zones for the project
func (co *CostOptimizer) getProjectZones() ([]string, error) {
	ctx := context.Background()
	zonesClient, err := compute.NewZonesRESTClient(ctx)
	if err != nil {
		return nil, err
	}
	defer zonesClient.Close()

	req := &computepb.ListZonesRequest{
		Project: co.projectID,
	}

	var zones []string
	it := zonesClient.List(ctx, req)
	for {
		zone, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		zones = append(zones, zone.GetName())
	}

	return zones, nil
}

// getResourceCost retrieves cost data from BigQuery
func (co *CostOptimizer) getResourceCost(resourceName, serviceName string) (float64, error) {
	ctx := context.Background()

	query := fmt.Sprintf(`
		SELECT 
			SUM(cost + IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) as monthly_cost
		FROM %s.%s.%s
		WHERE service.description = @service_name
			AND (resource.name LIKE @resource_pattern OR sku.description LIKE @resource_pattern)
			AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
	`, co.projectID, co.datasetID, co.tableID)

	q := co.bqClient.Query(query)
	q.Parameters = []bigquery.QueryParameter{
		{Name: "service_name", Value: serviceName},
		{Name: "resource_pattern", Value: "%" + resourceName + "%"},
	}

	it, err := q.Read(ctx)
	if err != nil {
		return 0, err
	}

	var row struct {
		MonthlyCost float64 `bigquery:"monthly_cost"`
	}

	err = it.Next(&row)
	if err != nil && err != iterator.Done {
		return 0, err
	}

	return row.MonthlyCost, nil
}

// getResourceMetrics retrieves utilization metrics from Cloud Monitoring
func (co *CostOptimizer) getResourceMetrics(zone, instanceName string) (ResourceMetrics, error) {
	ctx := context.Background()
	
	endTime := time.Now()
	startTime := endTime.Add(-24 * time.Hour) // Last 24 hours

	// Get CPU utilization
	cpuReq := &monitoringpb.ListTimeSeriesRequest{
		Name: fmt.Sprintf("projects/%s", co.projectID),
		Filter: fmt.Sprintf(`metric.type="compute.googleapis.com/instance/cpu/utilization" AND resource.labels.instance_name="%s" AND resource.labels.zone="%s"`, instanceName, zone),
		Interval: &monitoringpb.TimeInterval{
			EndTime:   &endTime,
			StartTime: &startTime,
		},
		View: monitoringpb.ListTimeSeriesRequest_FULL,
	}

	var cpuUtil, memUtil, diskUtil float64
	var networkBytes int64

	// Get CPU metrics
	cpuIt := co.monitoringClient.ListTimeSeries(ctx, cpuReq)
	cpuSeries, err := cpuIt.Next()
	if err != nil && err != iterator.Done {
		log.Printf("Error getting CPU metrics: %v", err)
	} else if cpuSeries != nil && len(cpuSeries.Points) > 0 {
		// Calculate average CPU utilization
		var total float64
		for _, point := range cpuSeries.Points {
			total += point.Value.GetDoubleValue()
		}
		cpuUtil = (total / float64(len(cpuSeries.Points))) * 100
	}

	// Get Memory utilization (if available)
	memReq := &monitoringpb.ListTimeSeriesRequest{
		Name: fmt.Sprintf("projects/%s", co.projectID),
		Filter: fmt.Sprintf(`metric.type="compute.googleapis.com/instance/memory/utilization" AND resource.labels.instance_name="%s" AND resource.labels.zone="%s"`, instanceName, zone),
		Interval: &monitoringpb.TimeInterval{
			EndTime:   &endTime,
			StartTime: &startTime,
		},
	}

	memIt := co.monitoringClient.ListTimeSeries(ctx, memReq)
	memSeries, err := memIt.Next()
	if err == nil && memSeries != nil && len(memSeries.Points) > 0 {
		var total float64
		for _, point := range memSeries.Points {
			total += point.Value.GetDoubleValue()
		}
		memUtil = (total / float64(len(memSeries.Points))) * 100
	}

	return ResourceMetrics{
		CPUUtilization:    cpuUtil,
		MemoryUtilization: memUtil,
		DiskUtilization:   diskUtil,
		NetworkBytes:      networkBytes,
		LastUpdated:       time.Now(),
	}, nil
}

// parseMachineType extracts vCPUs and memory from machine type name
func (co *CostOptimizer) parseMachineType(machineType string) (int64, int64) {
	// Machine type mapping (simplified)
	machineSpecs := map[string]struct{ vcpus, memoryMB int64 }{
		"e2-micro":    {1, 1024},
		"e2-small":    {1, 2048},
		"e2-medium":   {1, 4096},
		"e2-standard-2": {2, 8192},
		"e2-standard-4": {4, 16384},
		"e2-standard-8": {8, 32768},
		"n1-standard-1": {1, 3840},
		"n1-standard-2": {2, 7680},
		"n1-standard-4": {4, 15360},
		"n1-standard-8": {8, 30720},
		"n2-standard-2": {2, 8192},
		"n2-standard-4": {4, 16384},
		"n2-standard-8": {8, 32768},
		"c2-standard-4": {4, 16384},
		"c2-standard-8": {8, 32768},
		"c2-standard-16": {16, 65536},
	}

	if spec, exists := machineSpecs[machineType]; exists {
		return spec.vcpus, spec.memoryMB
	}

	// Parse custom machine types (e.g., custom-4-8192)
	if strings.HasPrefix(machineType, "custom-") {
		parts := strings.Split(machineType, "-")
		if len(parts) >= 3 {
			if vcpus, err := strconv.ParseInt(parts[1], 10, 64); err == nil {
				if memory, err := strconv.ParseInt(parts[2], 10, 64); err == nil {
					return vcpus, memory
				}
			}
		}
	}

	return 1, 1024 // Default fallback
}

// GenerateRecommendations analyzes resources and generates optimization recommendations
func (co *CostOptimizer) GenerateRecommendations(resources []Resource) ([]OptimizationRecommendation, error) {
	var recommendations []OptimizationRecommendation

	for _, resource := range resources {
		if resource.Type != "compute-instance" {
			continue
		}

		recs := co.analyzeComputeInstance(resource)
		recommendations = append(recommendations, recs...)
	}

	// Sort by potential savings
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].MonthlySavings > recommendations[j].MonthlySavings
	})

	return recommendations, nil
}

// analyzeComputeInstance generates recommendations for compute instances
func (co *CostOptimizer) analyzeComputeInstance(resource Resource) []OptimizationRecommendation {
	var recommendations []OptimizationRecommendation

	// Right-sizing recommendation based on CPU utilization
	if resource.Utilization.CPUUtilization > 0 {
		if resource.Utilization.CPUUtilization < 20 {
			// Suggest downsizing
			recommendedType, estimatedCost := co.suggestSmallerInstance(resource.MachineType, resource.CurrentCost)
			if recommendedType != resource.MachineType {
				savings := resource.CurrentCost - estimatedCost
				savingsPercent := (savings / resource.CurrentCost) * 100

				recommendations = append(recommendations, OptimizationRecommendation{
					ResourceID:      resource.ID,
					ResourceName:    resource.Name,
					ResourceType:    resource.Type,
					ProjectID:       resource.ProjectID,
					CurrentCost:     resource.CurrentCost,
					RecommendedType: recommendedType,
					EstimatedCost:   estimatedCost,
					MonthlySavings:  savings,
					SavingsPercent:  savingsPercent,
					Reason:          fmt.Sprintf("Low CPU utilization (%.1f%%). Instance is oversized.", resource.Utilization.CPUUtilization),
					Confidence:      "High",
					Impact:          "Low",
					RiskLevel:       "Low",
					AutoApplicable:  false,
				})
			}
		} else if resource.Utilization.CPUUtilization > 85 {
			// Suggest upsizing
			recommendedType, estimatedCost := co.suggestLargerInstance(resource.MachineType, resource.CurrentCost)
			if recommendedType != resource.MachineType {
				additionalCost := estimatedCost - resource.CurrentCost

				recommendations = append(recommendations, OptimizationRecommendation{
					ResourceID:      resource.ID,
					ResourceName:    resource.Name,
					ResourceType:    resource.Type,
					ProjectID:       resource.ProjectID,
					CurrentCost:     resource.CurrentCost,
					RecommendedType: recommendedType,
					EstimatedCost:   estimatedCost,
					MonthlySavings:  -additionalCost, // Negative savings (additional cost)
					SavingsPercent:  -(additionalCost / resource.CurrentCost) * 100,
					Reason:          fmt.Sprintf("High CPU utilization (%.1f%%). Performance may be impacted.", resource.Utilization.CPUUtilization),
					Confidence:      "Medium",
					Impact:          "Medium",
					RiskLevel:       "Medium",
					AutoApplicable:  false,
				})
			}
		}
	}

	// Idle resource detection
	if resource.Status == "RUNNING" && resource.Utilization.CPUUtilization < 5 && 
	   time.Since(resource.Utilization.LastUpdated) < 24*time.Hour {
		recommendations = append(recommendations, OptimizationRecommendation{
			ResourceID:      resource.ID,
			ResourceName:    resource.Name,
			ResourceType:    resource.Type,
			ProjectID:       resource.ProjectID,
			CurrentCost:     resource.CurrentCost,
			RecommendedType: "TERMINATED",
			EstimatedCost:   0,
			MonthlySavings:  resource.CurrentCost,
			SavingsPercent:  100,
			Reason:          "Instance appears to be idle with very low CPU utilization.",
			Confidence:      "Medium",
			Impact:          "High",
			RiskLevel:       "High",
			AutoApplicable:  false,
		})
	}

	// Preemptible instance recommendation
	if !strings.Contains(resource.Name, "preemptible") && resource.Utilization.CPUUtilization < 60 {
		preemptibleCost := resource.CurrentCost * 0.2 // Preemptible instances are ~80% cheaper
		savings := resource.CurrentCost - preemptibleCost

		recommendations = append(recommendations, OptimizationRecommendation{
			ResourceID:      resource.ID,
			ResourceName:    resource.Name,
			ResourceType:    resource.Type,
			ProjectID:       resource.ProjectID,
			CurrentCost:     resource.CurrentCost,
			RecommendedType: resource.MachineType + " (Preemptible)",
			EstimatedCost:   preemptibleCost,
			MonthlySavings:  savings,
			SavingsPercent:  (savings / resource.CurrentCost) * 100,
			Reason:          "Workload suitable for preemptible instances based on usage patterns.",
			Confidence:      "Medium",
			Impact:          "Low",
			RiskLevel:       "Medium",
			AutoApplicable:  false,
		})
	}

	// Committed Use Discount recommendation
	age := time.Since(resource.CreatedAt)
	if age > 30*24*time.Hour && resource.Status == "RUNNING" { // Running for more than 30 days
		cudSavings := resource.CurrentCost * 0.3 // 30% savings with 1-year commitment
		
		recommendations = append(recommendations, OptimizationRecommendation{
			ResourceID:      resource.ID,
			ResourceName:    resource.Name,
			ResourceType:    resource.Type,
			ProjectID:       resource.ProjectID,
			CurrentCost:     resource.CurrentCost,
			RecommendedType: resource.MachineType + " (CUD)",
			EstimatedCost:   resource.CurrentCost - cudSavings,
			MonthlySavings:  cudSavings,
			SavingsPercent:  30,
			Reason:          "Long-running instance eligible for Committed Use Discounts.",
			Confidence:      "High",
			Impact:          "Low",
			RiskLevel:       "Low",
			AutoApplicable:  true,
		})
	}

	return recommendations
}

// suggestSmallerInstance recommends a smaller machine type
func (co *CostOptimizer) suggestSmallerInstance(currentType string, currentCost float64) (string, float64) {
	downsizeMap := map[string]struct{ 
		recommended string
		costRatio   float64
	}{
		"n1-standard-8": {"n1-standard-4", 0.5},
		"n1-standard-4": {"n1-standard-2", 0.5},
		"n1-standard-2": {"n1-standard-1", 0.5},
		"n2-standard-8": {"n2-standard-4", 0.5},
		"n2-standard-4": {"n2-standard-2", 0.5},
		"e2-standard-8": {"e2-standard-4", 0.5},
		"e2-standard-4": {"e2-standard-2", 0.5},
		"e2-standard-2": {"e2-small", 0.4},
		"e2-medium":     {"e2-small", 0.5},
		"e2-small":      {"e2-micro", 0.5},
	}

	if suggestion, exists := downsizeMap[currentType]; exists {
		return suggestion.recommended, currentCost * suggestion.costRatio
	}

	return currentType, currentCost
}

// suggestLargerInstance recommends a larger machine type
func (co *CostOptimizer) suggestLargerInstance(currentType string, currentCost float64) (string, float64) {
	upsizeMap := map[string]struct{
		recommended string
		costRatio   float64
	}{
		"e2-micro":      {"e2-small", 2.0},
		"e2-small":      {"e2-medium", 2.0},
		"e2-medium":     {"e2-standard-2", 2.5},
		"e2-standard-2": {"e2-standard-4", 2.0},
		"e2-standard-4": {"e2-standard-8", 2.0},
		"n1-standard-1": {"n1-standard-2", 2.0},
		"n1-standard-2": {"n1-standard-4", 2.0},
		"n1-standard-4": {"n1-standard-8", 2.0},
		"n2-standard-2": {"n2-standard-4", 2.0},
		"n2-standard-4": {"n2-standard-8", 2.0},
	}

	if suggestion, exists := upsizeMap[currentType]; exists {
		return suggestion.recommended, currentCost * suggestion.costRatio
	}

	return currentType, currentCost
}

// GetCostAnalysis provides comprehensive cost analysis
func (co *CostOptimizer) GetCostAnalysis() (*CostAnalysis, error) {
	ctx := context.Background()

	query := fmt.Sprintf(`
		WITH monthly_costs AS (
			SELECT 
				project.id as project_id,
				service.description as service_name,
				location.location as location,
				DATE_TRUNC(DATE(usage_start_time), MONTH) as cost_month,
				SUM(cost + IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) as net_cost
			FROM %s.%s.%s
			WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
			GROUP BY project_id, service_name, location, cost_month
		),
		current_month AS (
			SELECT 
				project_id,
				service_name,
				location,
				SUM(net_cost) as current_cost
			FROM monthly_costs
			WHERE cost_month = DATE_TRUNC(CURRENT_DATE(), MONTH)
			GROUP BY project_id, service_name, location
		),
		previous_month AS (
			SELECT 
				project_id,
				service_name,
				location,
				SUM(net_cost) as previous_cost
			FROM monthly_costs
			WHERE cost_month = DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
			GROUP BY project_id, service_name, location
		)
		SELECT 
			c.project_id,
			c.service_name,
			c.location,
			c.current_cost,
			IFNULL(p.previous_cost, 0) as previous_cost,
			((c.current_cost - IFNULL(p.previous_cost, 0)) / NULLIF(p.previous_cost, 0)) * 100 as mom_growth_pct
		FROM current_month c
		LEFT JOIN previous_month p ON c.project_id = p.project_id 
			AND c.service_name = p.service_name 
			AND c.location = p.location
		ORDER BY c.current_cost DESC
	`, co.projectID, co.datasetID, co.tableID)

	q := co.bqClient.Query(query)
	it, err := q.Read(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to execute cost analysis query: %v", err)
	}

	var costBreakdown []ServiceCost
	var totalCurrent, totalPrevious float64

	for {
		var row struct {
			ProjectID     string  `bigquery:"project_id"`
			ServiceName   string  `bigquery:"service_name"`
			Location      string  `bigquery:"location"`
			CurrentCost   float64 `bigquery:"current_cost"`
			PreviousCost  float64 `bigquery:"previous_cost"`
			MomGrowthPct  float64 `bigquery:"mom_growth_pct"`
		}

		err := it.Next(&row)
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, err
		}

		costBreakdown = append(costBreakdown, ServiceCost{
			ProjectID:      row.ProjectID,
			ServiceName:    row.ServiceName,
			Location:       row.Location,
			CurrentCost:    row.CurrentCost,
			PreviousCost:   row.PreviousCost,
			GrowthPercent:  row.MomGrowthPct,
		})

		totalCurrent += row.CurrentCost
		totalPrevious += row.PreviousCost
	}

	totalGrowth := ((totalCurrent - totalPrevious) / totalPrevious) * 100

	return &CostAnalysis{
		TotalCurrentCost:  totalCurrent,
		TotalPreviousCost: totalPrevious,
		GrowthPercent:     totalGrowth,
		ServiceBreakdown:  costBreakdown,
		LastUpdated:       time.Now(),
	}, nil
}

// CostAnalysis represents comprehensive cost analysis
type CostAnalysis struct {
	TotalCurrentCost  float64       `json:"total_current_cost"`
	TotalPreviousCost float64       `json:"total_previous_cost"`
	GrowthPercent     float64       `json:"growth_percent"`
	ServiceBreakdown  []ServiceCost `json:"service_breakdown"`
	LastUpdated       time.Time     `json:"last_updated"`
}

// ServiceCost represents cost breakdown by service
type ServiceCost struct {
	ProjectID     string  `json:"project_id"`
	ServiceName   string  `json:"service_name"`
	Location      string  `json:"location"`
	CurrentCost   float64 `json:"current_cost"`
	PreviousCost  float64 `json:"previous_cost"`
	GrowthPercent float64 `json:"growth_percent"`
}

// HTTP Handlers
func (co *CostOptimizer) handleDiscoverResources(w http.ResponseWriter, r *http.Request) {
	resources, err := co.DiscoverResources()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resources)
}

func (co *CostOptimizer) handleGetRecommendations(w http.ResponseWriter, r *http.Request) {
	resources, err := co.DiscoverResources()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	recommendations, err := co.GenerateRecommendations(resources)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(recommendations)
}

func (co *CostOptimizer) handleGetCostAnalysis(w http.ResponseWriter, r *http.Request) {
	analysis, err := co.GetCostAnalysis()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(analysis)
}

func (co *CostOptimizer) handleOptimizationReport(w http.ResponseWriter, r *http.Request) {
	resources, err := co.DiscoverResources()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	recommendations, err := co.GenerateRecommendations(resources)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	analysis, err := co.GetCostAnalysis()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Calculate potential savings
	var totalSavings float64
	var highConfidenceRecs, autoApplicableRecs int
	
	for _, rec := range recommendations {
		if rec.MonthlySavings > 0 {
			totalSavings += rec.MonthlySavings
		}
		if rec.Confidence == "High" {
			highConfidenceRecs++
		}
		if rec.AutoApplicable {
			autoApplicableRecs++
		}
	}

	report := map[string]interface{}{
		"summary": map[string]interface{}{
			"total_resources":          len(resources),
			"total_recommendations":    len(recommendations),
			"potential_monthly_savings": totalSavings,
			"savings_percentage":       (totalSavings / analysis.TotalCurrentCost) * 100,
			"high_confidence_recs":     highConfidenceRecs,
			"auto_applicable_recs":     autoApplicableRecs,
		},
		"cost_analysis":    analysis,
		"resources":        resources,
		"recommendations":  recommendations,
		"generated_at":     time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(report)
}

func main() {
	projectID := os.Getenv("GCP_PROJECT_ID")
	datasetID := os.Getenv("BQ_DATASET_ID")
	tableID := os.Getenv("BQ_TABLE_ID")
	port := os.Getenv("PORT")

	if projectID == "" || datasetID == "" || tableID == "" {
		log.Fatal("Environment variables GCP_PROJECT_ID, BQ_DATASET_ID, and BQ_TABLE_ID are required")
	}

	if port == "" {
		port = "8080"
	}

	optimizer, err := NewCostOptimizer(projectID, datasetID, tableID)
	if err != nil {
		log.Fatalf("Failed to initialize cost optimizer: %v", err)
	}
	defer optimizer.bqClient.Close()
	defer optimizer.computeClient.Close()
	defer optimizer.monitoringClient.Close()

	r := mux.NewRouter()
	r.HandleFunc("/api/resources", optimizer.handleDiscoverResources).Methods("GET")
	r.HandleFunc("/api/recommendations", optimizer.handleGetRecommendations).Methods("GET")
	r.HandleFunc("/api/cost-analysis", optimizer.handleGetCostAnalysis).Methods("GET")
	r.HandleFunc("/api/optimization-report", optimizer.handleOptimizationReport).Methods("GET")

	// Health check endpoint
	r.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	}).Methods("GET")

	log.Printf("Starting Cost Optimizer API server on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, r))
}