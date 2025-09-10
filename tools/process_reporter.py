"""
Process Reporter - Generates comprehensive reports analyzing agent outputs
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sqlite3
from pathlib import Path


@dataclass
class AgentContribution:
    """Represents an agent's contribution to the pipeline"""
    name: str
    messages: List[str]
    key_outputs: List[str]
    errors: List[str]
    warnings: List[str]
    execution_time: float
    token_usage: int
    performance_metrics: Dict[str, Any]


class ProcessReporter:
    """Generates comprehensive process reports by analyzing agent communications"""
    
    def __init__(self, job_id: str, db_path: str = "automl_system.db"):
        self.job_id = job_id
        self.db_path = db_path
        self.agent_contributions = {}
        
    def analyze_agent_messages(self) -> Dict[str, AgentContribution]:
        """Analyze all agent messages for the job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all messages
        cursor.execute("""
            SELECT agent_name, content, message_type, timestamp
            FROM agent_messages 
            WHERE job_id = ? 
            ORDER BY timestamp ASC
        """, (self.job_id,))
        
        messages = cursor.fetchall()
        
        # Get agent statistics
        cursor.execute("""
            SELECT agent_name, tokens_consumed, calls_count, total_execution_time
            FROM agent_statistics 
            WHERE job_id = ?
        """, (self.job_id,))
        
        stats = cursor.fetchall()
        conn.close()
        
        # Group by agent
        agent_data = {}
        for msg in messages:
            agent_name, content, msg_type, timestamp = msg
            if agent_name not in agent_data:
                agent_data[agent_name] = {
                    'messages': [],
                    'key_outputs': [],
                    'errors': [],
                    'warnings': [],
                    'stats': {'tokens': 0, 'calls': 0, 'time': 0}
                }
            
            agent_data[agent_name]['messages'].append(content)
            
            # Extract key information
            if 'ERROR_START:' in content:
                error = self._extract_between_markers(content, 'ERROR_START:', ':ERROR_END')
                if error:
                    agent_data[agent_name]['errors'].append(error)
                    
            if 'LOG_START:' in content:
                log = self._extract_between_markers(content, 'LOG_START:', 'LOG_END:')
                if log and 'WARNING' not in content:
                    agent_data[agent_name]['key_outputs'].append(log)
                    
            if 'WARNING' in content.upper():
                agent_data[agent_name]['warnings'].append(content)
                
            # Extract structured outputs
            if 'MODEL_PATH_START:' in content:
                model_path = self._extract_between_markers(content, 'MODEL_PATH_START:', ':MODEL_PATH_END')
                if model_path:
                    agent_data[agent_name]['key_outputs'].append(f"Model saved: {model_path}")
                    
            if 'METRICS_START:' in content:
                metrics = self._extract_between_markers(content, 'METRICS_START:', ':METRICS_END')
                if metrics:
                    agent_data[agent_name]['key_outputs'].append(f"Metrics generated: {metrics[:100]}...")
        
        # Add statistics
        stats_dict = {stat[0]: {'tokens': stat[1], 'calls': stat[2], 'time': stat[3]} for stat in stats}
        
        # Create AgentContribution objects
        contributions = {}
        for agent_name, data in agent_data.items():
            stat = stats_dict.get(agent_name, {'tokens': 0, 'calls': 0, 'time': 0})
            contributions[agent_name] = AgentContribution(
                name=agent_name,
                messages=data['messages'],
                key_outputs=data['key_outputs'],
                errors=data['errors'],
                warnings=data['warnings'],
                execution_time=stat['time'],
                token_usage=stat['tokens'],
                performance_metrics=stat
            )
            
        return contributions
    
    def _extract_between_markers(self, text: str, start: str, end: str) -> Optional[str]:
        """Extract text between markers"""
        start_idx = text.find(start)
        if start_idx == -1:
            return None
            
        start_idx += len(start)
        end_idx = text.find(end, start_idx)
        if end_idx == -1:
            return None
            
        return text[start_idx:end_idx].strip()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive process report"""
        contributions = self.analyze_agent_messages()
        
        # Get job information
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (self.job_id,))
        job_data = cursor.fetchone()
        
        cursor.execute("SELECT * FROM models WHERE job_id = ?", (self.job_id,))
        models = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) FROM agent_messages WHERE job_id = ?", (self.job_id,))
        total_messages = cursor.fetchone()[0]
        
        conn.close()
        
        # Analyze pipeline performance
        total_execution_time = sum(contrib.execution_time for contrib in contributions.values())
        total_tokens = sum(contrib.token_usage for contrib in contributions.values())
        
        # Generate insights
        insights = self._generate_insights(contributions, job_data, models)
        
        # Create comprehensive report
        report = {
            "job_id": self.job_id,
            "job_name": job_data[1] if job_data else "Unknown",
            "generated_at": datetime.now().isoformat(),
            
            "executive_summary": insights["executive_summary"],
            
            "pipeline_overview": {
                "total_execution_time": total_execution_time,
                "total_messages_exchanged": total_messages,
                "total_tokens_consumed": total_tokens,
                "agents_involved": len(contributions),
                "models_generated": len(models) if models else 0,
                "learning_type": insights["learning_type"],
                "target_column": job_data[9] if job_data else None
            },
            
            "agent_analysis": self._analyze_agent_performance(contributions),
            
            "process_flow": self._analyze_process_flow(contributions),
            
            "key_achievements": insights["achievements"],
            
            "challenges_and_resolutions": insights["challenges"],
            
            "performance_metrics": {
                "efficiency_score": self._calculate_efficiency_score(contributions),
                "success_rate": self._calculate_success_rate(contributions),
                "collaboration_quality": self._assess_collaboration_quality(contributions)
            },
            
            "technical_details": {
                "data_processing": insights["data_processing"],
                "model_training": insights["model_training"],
                "predictions": insights["predictions"],
                "visualizations": insights["visualizations"]
            },
            
            "recommendations": insights["recommendations"]
        }
        
        return report
    
    def _generate_insights(self, contributions: Dict[str, AgentContribution], job_data, models) -> Dict[str, Any]:
        """Generate insights from agent contributions"""
        insights = {
            "executive_summary": "",
            "learning_type": "supervised",
            "achievements": [],
            "challenges": [],
            "data_processing": {},
            "model_training": {},
            "predictions": {},
            "visualizations": {},
            "recommendations": []
        }
        
        # Determine learning type
        target_column = job_data[9] if job_data else ""
        if not target_column or target_column.strip() == "":
            insights["learning_type"] = "unsupervised"
        
        # Analyze each agent's contribution
        for agent_name, contrib in contributions.items():
            
            if agent_name == "DataProcessorAgent":
                insights["data_processing"] = {
                    "messages_count": len(contrib.messages),
                    "key_findings": contrib.key_outputs[:3],  # Top 3 findings
                    "data_quality_issues": len(contrib.warnings)
                }
                
            elif agent_name == "ModelBuilderAgent":
                insights["model_training"] = {
                    "attempts": len(contrib.messages),
                    "errors_encountered": len(contrib.errors),
                    "final_model_details": contrib.key_outputs[-1] if contrib.key_outputs else "None",
                    "training_approach": "Isolation Forest" if insights["learning_type"] == "unsupervised" else "H2O AutoML"
                }
                
            elif agent_name == "PredictionAgent":
                insights["predictions"] = {
                    "prediction_tasks": len(contrib.key_outputs),
                    "success": len(contrib.errors) == 0,
                    "details": contrib.key_outputs
                }
                
            elif agent_name == "VisualizationAgent":
                insights["visualizations"] = {
                    "charts_generated": len(contrib.key_outputs),
                    "visualization_type": "Anomaly Detection Charts" if insights["learning_type"] == "unsupervised" else "Forecast Plots",
                    "success": len(contrib.errors) == 0
                }
        
        # Generate executive summary
        if insights["learning_type"] == "unsupervised":
            insights["executive_summary"] = f"Unsupervised anomaly detection pipeline completed successfully. {len(models)} models trained for fraud detection using Isolation Forest algorithm."
        else:
            insights["executive_summary"] = f"Supervised machine learning pipeline completed. {len(models)} models trained using H2O AutoML for prediction task."
        
        # Generate achievements
        if models:
            insights["achievements"].append(f"Successfully trained {len(models)} machine learning model(s)")
        
        if not any(contrib.errors for contrib in contributions.values()):
            insights["achievements"].append("Pipeline completed without critical errors")
            
        # Generate challenges
        total_errors = sum(len(contrib.errors) for contrib in contributions.values())
        if total_errors > 0:
            insights["challenges"].append(f"Resolved {total_errors} errors during execution")
            
        # Generate recommendations
        if insights["learning_type"] == "unsupervised":
            insights["recommendations"].append("Review flagged anomalies for potential fraud investigation")
            insights["recommendations"].append("Consider threshold tuning based on business requirements")
        else:
            insights["recommendations"].append("Validate model performance on new data")
            insights["recommendations"].append("Monitor model drift over time")
            
        return insights
    
    def _analyze_agent_performance(self, contributions: Dict[str, AgentContribution]) -> Dict[str, Any]:
        """Analyze individual agent performance"""
        analysis = {}
        
        for agent_name, contrib in contributions.items():
            analysis[agent_name] = {
                "efficiency": {
                    "messages_per_minute": len(contrib.messages) / max(contrib.execution_time / 60, 1),
                    "tokens_per_message": contrib.token_usage / max(len(contrib.messages), 1),
                    "error_rate": len(contrib.errors) / max(len(contrib.messages), 1)
                },
                "contributions": {
                    "total_messages": len(contrib.messages),
                    "key_outputs": len(contrib.key_outputs),
                    "errors_handled": len(contrib.errors),
                    "warnings_issued": len(contrib.warnings)
                },
                "performance_rating": self._calculate_agent_rating(contrib)
            }
            
        return analysis
    
    def _analyze_process_flow(self, contributions: Dict[str, AgentContribution]) -> List[Dict[str, Any]]:
        """Analyze the flow of the process"""
        # Order agents by typical pipeline flow
        agent_order = ["DataProcessorAgent", "ModelBuilderAgent", "CodeExecutorAgent", 
                      "AnalystAgent", "PredictionAgent", "VisualizationAgent"]
        
        flow = []
        for agent_name in agent_order:
            if agent_name in contributions:
                contrib = contributions[agent_name]
                flow.append({
                    "agent": agent_name,
                    "role": self._get_agent_role_description(agent_name),
                    "duration": contrib.execution_time,
                    "output_quality": "High" if len(contrib.errors) == 0 else "Medium" if len(contrib.errors) < 3 else "Low",
                    "key_deliverables": contrib.key_outputs[:2]  # Top 2 deliverables
                })
                
        return flow
    
    def _get_agent_role_description(self, agent_name: str) -> str:
        """Get role description for an agent"""
        roles = {
            "DataProcessorAgent": "Data analysis and preprocessing",
            "ModelBuilderAgent": "Machine learning model creation",
            "CodeExecutorAgent": "Script execution and validation",
            "AnalystAgent": "Quality assurance and validation",
            "PredictionAgent": "Prediction generation",
            "VisualizationAgent": "Chart and visualization creation"
        }
        return roles.get(agent_name, "General processing")
    
    def _calculate_efficiency_score(self, contributions: Dict[str, AgentContribution]) -> float:
        """Calculate overall pipeline efficiency score (0-100)"""
        if not contributions:
            return 0
            
        total_messages = sum(len(contrib.messages) for contrib in contributions.values())
        total_errors = sum(len(contrib.errors) for contrib in contributions.values())
        
        error_rate = total_errors / max(total_messages, 1)
        efficiency = max(0, 100 - (error_rate * 50))  # 50 points penalty for each error rate point
        
        return round(efficiency, 2)
    
    def _calculate_success_rate(self, contributions: Dict[str, AgentContribution]) -> float:
        """Calculate success rate based on outputs vs errors"""
        total_outputs = sum(len(contrib.key_outputs) for contrib in contributions.values())
        total_errors = sum(len(contrib.errors) for contrib in contributions.values())
        
        if total_outputs + total_errors == 0:
            return 0
            
        success_rate = (total_outputs / (total_outputs + total_errors)) * 100
        return round(success_rate, 2)
    
    def _assess_collaboration_quality(self, contributions: Dict[str, AgentContribution]) -> str:
        """Assess the quality of collaboration between agents"""
        total_agents = len(contributions)
        agents_with_outputs = sum(1 for contrib in contributions.values() if len(contrib.key_outputs) > 0)
        
        if total_agents == 0:
            return "No Data"
        
        participation_rate = agents_with_outputs / total_agents
        
        if participation_rate >= 0.8:
            return "Excellent"
        elif participation_rate >= 0.6:
            return "Good"
        elif participation_rate >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_agent_rating(self, contrib: AgentContribution) -> str:
        """Calculate performance rating for an agent"""
        if len(contrib.messages) == 0:
            return "No Activity"
            
        output_ratio = len(contrib.key_outputs) / len(contrib.messages)
        error_ratio = len(contrib.errors) / len(contrib.messages)
        
        score = output_ratio - error_ratio
        
        if score >= 0.3:
            return "Excellent"
        elif score >= 0.15:
            return "Good"
        elif score >= 0:
            return "Fair"
        else:
            return "Needs Improvement"


def generate_process_report(job_id: str, db_path: str = "automl_system.db") -> Dict[str, Any]:
    """Generate a comprehensive process report for a job"""
    reporter = ProcessReporter(job_id, db_path)
    return reporter.generate_comprehensive_report()