# ☁️ Multi-Cloud Infrastructure Cost Optimizer

An intelligent, cross-cloud cost optimization system designed to analyze cloud resource usage, identify inefficiencies, and recommend actionable strategies to reduce cloud spending.

> 🔧 Built with **Go**, **Python**, **GCP (BigQuery, Compute, Monitoring)**, **FAISS**, **ML models**, **Docker**, and **REST APIs**

---

## 📌 Key Features

- 🔍 **Automated Resource Discovery**: Scans GCP infrastructure for compute instances across all zones.
- 📉 **Cost Analysis & Trend Monitoring**: Queries BigQuery billing exports to extract 6-month cost trends.
- 📈 **Utilization-Aware Recommendations**: Uses metrics from Cloud Monitoring to detect under/over-utilized VMs.
- 💡 **ML-Driven Optimization**: Python modules suggest instance right-sizing, preemptible migration, idle shutdown, and CUD opportunities.
- 📊 **Savings Reports**: Generates real-time reports with projected monthly savings and confidence levels.
- 🧠 **AI-Enabled Insights**: FAISS-powered embeddings (future-ready) to enhance multi-modal optimization via semantic retrieval.

---

## 🗃️ Project Structure

Multi-Cloud-Infrastructure-Cost-Optimizer/
├── Main Service.go # Main Go backend service for resource discovery, cost analysis, APIs
├── Automation and ML Components.py # Python logic for ML-based recommendation and integration
├── Dockerfile # Container config (to be added)
├── README.md # Project documentation
├── .env.example # Example environment variables
└── (Add infra/ for Terraform, deploy/ for Cloud Run configs if needed)

yaml
Copy
Edit

---

## ⚙️ Setup & Usage

### 1. Prerequisites

- Go 1.20+
- Python 3.8+
- GCP Billing Export set up in BigQuery
- Enable APIs: BigQuery, Compute Engine, Cloud Monitoring

### 2. Set Environment Variables

Create `.env` file:

GCP_PROJECT_ID=your-gcp-project
BQ_DATASET_ID=your_bq_dataset
BQ_TABLE_ID=your_bq_table
PORT=8080

r
Copy
Edit

### 3. Run Go Backend (API Server)

```bash
go mod tidy
go run Main\ Service.go
Endpoints:

/api/resources → List all compute instances with metrics

/api/recommendations → Get cost-saving recommendations

/api/cost-analysis → Month-over-month cost breakdown

/api/optimization-report → Unified report with summary + actions

/health → Health check

4. Run Python ML Modules
bash
Copy
Edit
python3 "Automation and ML Components.py"
💻 Technologies Used
Layer	Tech/Service
Language	Go, Python
Cloud Platform	Google Cloud (GCP)
Billing Analytics	BigQuery
Metrics/Monitoring	Cloud Monitoring API
Resource Discovery	Compute Engine API
ML Components	Scikit-learn, FAISS
Containerization	Docker (optional for deployment)
Deployment	Cloud Run / Localhost

🧠 Sample Recommendations
💡 Downsize oversized instances based on low CPU/memory utilization

💰 Replace with preemptible VMs to save up to 80%

🔌 Identify idle resources for termination

🕐 Recommend Committed Use Discounts (CUDs) after 30+ days of use

📈 Detect performance bottlenecks needing upsizing

📊 Sample Optimization Report Output (JSON)
json
Copy
Edit
{
  "summary": {
    "total_resources": 42,
    "total_recommendations": 12,
    "potential_monthly_savings": 195.30,
    "savings_percentage": 21.8,
    "high_confidence_recs": 8,
    "auto_applicable_recs": 5
  },
  "cost_analysis": {...},
  "resources": [...],
  "recommendations": [...],
  "generated_at": "2025-07-26T12:00:00Z"
}
📦 Future Scope
🟡 Multi-cloud support (AWS, Azure)

🟣 Natural Language Query Interface using LangChain

🔵 Alerts & Scheduled Jobs via Cloud Scheduler + Pub/Sub

🔴 CI/CD integration with GitHub Actions + Terraform IaC

📃 License
MIT License. See LICENSE for more information.
