// Global variables
let currentJobId = null;
// Chart variable removed
let currentSection = 'dashboard';
let previousSection = null;

// Connection editing variables
let isEditingConnection = false;
let currentEditingConnectionId = null;

// Dark mode functionality
function initDarkMode() {
    // Check for saved dark mode preference or default to light mode
    const darkMode = localStorage.getItem('darkMode') === 'true';
    if (darkMode) {
        document.documentElement.classList.add('dark');
        updateThemeIcon('dark');
    } else {
        document.documentElement.classList.remove('dark');
        updateThemeIcon('light');
    }
}

function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', isDark);
    updateThemeIcon(isDark ? 'dark' : 'light');
    
    // Chart functionality removed
}

function updateThemeIcon(theme) {
    const icon = document.getElementById('theme-icon');
    const highlightTheme = document.getElementById('highlight-theme');
    
    if (theme === 'dark') {
        icon.className = 'fas fa-sun text-lg';
        highlightTheme.href = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css';
    } else {
        icon.className = 'fas fa-moon text-lg';
        highlightTheme.href = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css';
    }
}

function updateLastUpdated() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    const element = document.getElementById('last-updated');
    if (element) {
        element.textContent = `Last updated: ${timeString}`;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initDarkMode();
    
    // Initialize highlight.js if available
    if (typeof hljs !== 'undefined') {
        hljs.configure({
            languages: ['python', 'javascript', 'json', 'sql', 'bash', 'shell'],
            ignoreUnescapedHTML: true
        });
    }
    
    showSection('dashboard');
    loadDashboard();
    
    // Set up form submission
    document.getElementById('job-form').addEventListener('submit', createJob);
    
    // Set up navigation event listeners
    document.getElementById('nav-dashboard').addEventListener('click', (e) => showSection('dashboard', e));
    document.getElementById('nav-create-job').addEventListener('click', (e) => showSection('create-job', e));
    document.getElementById('nav-jobs').addEventListener('click', (e) => showSection('jobs', e));
    document.getElementById('nav-dataset-sql').addEventListener('click', (e) => showSection('dataset-sql', e));
    document.getElementById('nav-configuration').addEventListener('click', (e) => showSection('configuration', e));
    const chatContainer = document.getElementById('job-chat-messages');
    if (chatContainer) {
        chatContainer.addEventListener('scroll', handleChatScroll);
    }

});

function handleChatScroll() {
    const chatContainer = document.getElementById('job-chat-messages');
    const scrollButton = document.getElementById('scroll-to-bottom');

    if (chatContainer && scrollButton) {
        // Si el usuario hace scroll hasta el final, oculta el botón.
        if (chatContainer.scrollTop + chatContainer.clientHeight >= chatContainer.scrollHeight - 10) {
            scrollButton.classList.add('opacity-0', 'invisible');
        }
    }
}

// Navigation functions
function showSection(sectionName, event = null) {
    // Hide all sections
    const sections = ['dashboard-section', 'create-job-section', 'jobs-section', 'job-details-section', 'dataset-sql-section', 'configuration-section', 'process-report-section'];
    sections.forEach(section => {
        document.getElementById(section).classList.add('hidden');
    });
    
    // Show selected section
    document.getElementById(sectionName + '-section').classList.remove('hidden');
    
    // Store previous section for navigation
    if (currentSection !== sectionName) {
        previousSection = currentSection;
        currentSection = sectionName;
    }
    
    // Update nav buttons (only for main navigation, not job details)
    if (sectionName !== 'job-details') {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('bg-white/30');
            btn.classList.add('bg-white/20');
        });
        
        // Only update button styling if called from a click event
        if (event && event.target) {
            event.target.classList.remove('bg-white/20');
            event.target.classList.add('bg-white/30');
        }
    }
    
    // Load section-specific data
    if (sectionName === 'dashboard') {
        loadDashboard();
    } else if (sectionName === 'jobs') {
        loadJobs();
    } else if (sectionName === 'create-job') {
        loadSqlDatasetOptions();
    } else if (sectionName === 'dataset-sql') {
        loadSqlDatasets();
        loadDbConnections();
    } else if (sectionName === 'configuration') {
        loadConfiguration();
    }
}

// Navigation helper functions
function goBack() {
    if (previousSection) {
        showSection(previousSection);
        stopChatAutoRefresh();
    } else {
        showSection('jobs');
    }
}

function openJobDetails(jobId) {
    currentJobId = jobId;
    showSection('job-details');
    loadJobDetails(jobId);
}

// Dashboard functions
async function loadDashboard() {
    try {
        const response = await fetch('/jobs');
        const jobs = await response.json();
        
        updateDashboardStats(jobs);
        updateRecentJobs(jobs.slice(0, 5));
        updateLastUpdated();
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showNotification('Error loading dashboard data', 'error');
    }
}

function updateDashboardStats(jobs) {
    const totalJobs = jobs.length;
    const completedJobs = jobs.filter(job => job.status === 'completed').length;
    const runningJobs = jobs.filter(job => ['processing', 'analyzing_data', 'training_model', 'generating_predictions'].includes(job.status)).length;
    const waitingJobs = jobs.filter(job => job.status === 'awaiting_user_input').length;
    const failedJobs = jobs.filter(job => job.status === 'failed').length;
    
    document.getElementById('total-jobs').textContent = totalJobs;
    document.getElementById('completed-jobs').textContent = completedJobs;
    document.getElementById('running-jobs').textContent = runningJobs + (waitingJobs > 0 ? ` (${waitingJobs} waiting for input)` : '');
    document.getElementById('failed-jobs').textContent = failedJobs;
}

// Chart functionality removed - updateJobStatusChart function deleted

function updateRecentJobs(jobs) {
    const recentJobsContainer = document.getElementById('recent-jobs');
    
    if (jobs.length === 0) {
        recentJobsContainer.innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-inbox text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                <p class="text-gray-500 dark:text-gray-400">No jobs found</p>
                <button onclick="showSection('create-job')" class="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm">
                    Create your first job
                </button>
            </div>
        `;
        return;
    }
    
    recentJobsContainer.innerHTML = jobs.map(job => `
        <div class="border border-gray-200 dark:border-dark-600 rounded-lg p-4 hover:shadow-lg hover:bg-gray-50 dark:hover:bg-dark-700 transition-all duration-300 cursor-pointer transform hover:-translate-y-1" onclick="openJobDetails('${job.id}')">
            <div class="flex justify-between items-start">
                <div class="flex-1 min-w-0">
                    <div class="flex items-center mb-2">
                        <h4 class="font-semibold text-gray-800 dark:text-white truncate mr-2">${job.name}</h4>
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusBadgeClass(job.status)} shrink-0">
                            ${getStatusIcon(job.status)} ${job.status}
                        </span>
                    </div>
                    <p class="text-gray-600 dark:text-gray-400 text-sm mb-2 line-clamp-2">${job.prompt.substring(0, 120)}${job.prompt.length > 120 ? '...' : ''}</p>
                    <div class="flex items-center text-xs text-gray-500 dark:text-gray-400 space-x-3">
                        <span><i class="fas fa-calendar mr-1"></i>${formatDate(job.created_at)}</span>
                        <span><i class="fas fa-target mr-1"></i>${job.target_column}</span>
                    </div>
                </div>
                <div class="ml-4 text-right">
                    <div class="mb-2">
                        <div class="w-24 bg-gray-200 dark:bg-dark-600 rounded-full h-2 mb-1">
                            <div class="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500" style="width: ${job.progress}%"></div>
                        </div>
                        <span class="text-xs text-gray-500 dark:text-gray-400">${job.progress}%</span>
                    </div>
                    ${job.status === 'failed' ? `
                    <button onclick="retryJob('${job.id}'); event.stopPropagation();" class="bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white px-2 py-1 rounded text-xs transition-all duration-300 mr-1 mb-1">
                        <i class="fas fa-redo mr-1"></i>Retry
                    </button>
                    ` : ''}
                    
                    <!-- Button to create new version (available for all) -->
                    <button onclick="createVersionFromJob('${job.id}'); event.stopPropagation();" class="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-2 py-1 rounded text-xs transition-all duration-300 mr-1 mb-1" title="Create new version">
                        <i class="fas fa-plus mr-1"></i>New Version
                    </button>
                    
                    ${job.is_parent ? `
                    <!-- Button to view child versions -->
                    <button onclick="showChildVersions('${job.id}'); event.stopPropagation();" class="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white px-2 py-1 rounded text-xs transition-all duration-300 mr-1 mb-1" title="View child versions">
                        <i class="fas fa-code-branch mr-1"></i>Children (v1)
                    </button>
                    ` : `
                    <!-- Button to view parent version -->
                    <button onclick="showParentVersion('${job.parent_job_id}'); event.stopPropagation();" class="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white px-2 py-1 rounded text-xs transition-all duration-300 mr-1 mb-1" title="View parent version">
                        <i class="fas fa-arrow-up mr-1"></i>Parent
                    </button>
                    <!-- Button to view siblings -->
                    <button onclick="showSiblingVersions('${job.parent_job_id}', '${job.id}'); event.stopPropagation();" class="bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600 text-white px-2 py-1 rounded text-xs transition-all duration-300 mr-1 mb-1" title="View all versions">
                        <i class="fas fa-sitemap mr-1"></i>Siblings (v${job.version_number})
                    </button>
                    `}
                </div>
            </div>
        </div>
    `).join('');
}

// Job creation functions
async function createJob(event) {
    event.preventDefault();
    
    const name = document.getElementById('job-name').value;
    const prompt = document.getElementById('job-prompt').value;
    const targetColumn = document.getElementById('target-column').value;
    
    // Check which dataset source is selected
    const fileInput = document.getElementById('dataset-file');
    const sqlDatasetSelect = document.getElementById('sql-dataset-select');
    
    const fileSection = document.getElementById('file-upload-section');
    const sqlSection = document.getElementById('sql-dataset-section');
    
    const isFileSelected = !fileSection.classList.contains('hidden') && fileInput.files[0];
    const isSqlDatasetSelected = !sqlSection.classList.contains('hidden') && sqlDatasetSelect.value;
    
    if (!isFileSelected && !isSqlDatasetSelected) {
        showNotification('Please select a dataset file or SQL dataset', 'error');
        return;
    }
    
    try {
        // Create job
        const jobResponse = await fetch('/jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                prompt: prompt,
                target_column: targetColumn
            }),
        });
        
        const jobResult = await jobResponse.json();
        
        if (!jobResponse.ok) {
            throw new Error(jobResult.detail || 'Failed to create job');
        }
        
        if (isFileSelected) {
            // Upload file dataset
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const uploadResponse = await fetch(`/jobs/${jobResult.job_id}/upload`, {
                method: 'POST',
                body: formData,
            });
            
            if (!uploadResponse.ok) {
                throw new Error('Failed to upload dataset');
            }
        } else if (isSqlDatasetSelected) {
            // Use SQL dataset
            const datasetId = sqlDatasetSelect.value;
            
            const sqlDatasetResponse = await fetch(`/jobs/${jobResult.job_id}/use-sql-dataset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset_id: datasetId
                }),
            });
            
            if (!sqlDatasetResponse.ok) {
                const error = await sqlDatasetResponse.json();
                throw new Error(error.detail || 'Failed to use SQL dataset');
            }
        }
        
        showNotification('Job created successfully!', 'success');
        resetForm();
        showSection('jobs');
        loadJobs();
        
    } catch (error) {
        console.error('Error creating job:', error);
        showNotification(error.message, 'error');
    }
}

function resetForm() {
    document.getElementById('job-form').reset();
}

// Jobs list functions
async function loadJobs() {
    try {
        const response = await fetch('/jobs');
        const jobs = await response.json();
        
        displayJobs(jobs);
    } catch (error) {
        console.error('Error loading jobs:', error);
        showNotification('Error loading jobs', 'error');
    }
}

function displayJobs(jobs) {
    const jobsContainer = document.getElementById('jobs-list');
    
    if (jobs.length === 0) {
        jobsContainer.innerHTML = '<p class="text-gray-500 text-center">No jobs found. Create your first job!</p>';
        return;
    }
    
    jobsContainer.innerHTML = jobs.map(job => `
        <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-6 hover:shadow-md dark:bg-gray-900" id="job-${job.id}">
            <div class="flex justify-between items-start mb-4">
                <div class="flex-1">
                    <div class="mb-2">
                        <h3 id="job-name-display-${job.id}" class="text-xl font-semibold text-gray-900 dark:text-white">${job.name}</h3>
                        <div id="job-name-edit-${job.id}" class="hidden">
                            <input type="text" id="job-name-input-${job.id}" value="${job.name}" 
                                class="text-xl font-semibold bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 w-full max-w-md">
                            <div class="flex space-x-2 mt-2">
                                <button onclick="saveJobName('${job.id}')" class="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded text-sm">
                                    <i class="fas fa-check mr-1"></i>Save
                                </button>
                                <button onclick="cancelEditJobName('${job.id}')" class="bg-gray-500 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm">
                                    <i class="fas fa-times mr-1"></i>Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                    <p class="text-gray-600 dark:text-gray-300 mb-2">${job.prompt}</p>
                    <div class="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                        <span><i class="fas fa-calendar mr-1"></i>Created: ${formatDate(job.created_at)}</span>
                        <span><i class="fas fa-target mr-1"></i>Target: ${job.target_column}</span>
                    </div>
                </div>
                <div class="flex flex-col items-end space-y-2">
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusBadgeClass(job.status)}">
                        ${job.status}
                    </span>
                    <div class="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${job.progress}%"></div>
                    </div>
                    <span class="text-xs text-gray-500 dark:text-gray-400">${job.progress}%</span>
                </div>
            </div>
            
            <div class="flex flex-wrap gap-2">
                <!-- Acciones principales -->
                <button onclick="openJobDetails('${job.id}')" class="bg-blue-500 hover:bg-blue-700 text-white px-3 py-2 rounded text-sm">
                    <i class="fas fa-eye mr-1"></i>View Details
                </button>
                <button onclick="editJobName('${job.id}')" class="bg-purple-500 hover:bg-purple-700 text-white px-3 py-2 rounded text-sm">
                    <i class="fas fa-edit mr-1"></i>Edit Name
                </button>
                
                <!-- Botones de versionado -->
                <button onclick="createVersionFromJob('${job.id}')" class="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-3 py-2 rounded text-sm" title="Create new version">
                    <i class="fas fa-plus mr-1"></i>New Version
                </button>
                
                ${job.is_parent ? `
                <button onclick="showChildVersions('${job.id}')" class="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white px-3 py-2 rounded text-sm" title="Ver versiones hijas">
                    <i class="fas fa-code-branch mr-1"></i>Children (v1)
                </button>
                ` : `
                <button onclick="showParentVersion('${job.parent_job_id}')" class="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white px-3 py-2 rounded text-sm" title="View parent version">
                    <i class="fas fa-arrow-up mr-1"></i>Parent
                </button>
                <button onclick="showSiblingVersions('${job.parent_job_id}', '${job.id}')" class="bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600 text-white px-3 py-2 rounded text-sm" title="Ver todas las versiones">
                    <i class="fas fa-sitemap mr-1"></i>Siblings (v${job.version_number})
                </button>
                `}
                
                <!-- Acciones adicionales -->
                <button onclick="downloadResults('${job.id}')" class="bg-green-500 hover:bg-green-700 text-white px-3 py-2 rounded text-sm" ${job.status !== 'completed' ? 'disabled' : ''}>
                    <i class="fas fa-download mr-1"></i>Download
                </button>
                <button onclick="showProcessReport('${job.id}')" class="bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 text-white px-3 py-2 rounded text-sm" title="Ver reporte de proceso de agentes">
                    <i class="fas fa-chart-line mr-1"></i>Process Report
                </button>
                ${job.status === 'failed' ? `
                <button onclick="retryJob('${job.id}')" class="bg-yellow-500 hover:bg-yellow-700 text-white px-3 py-2 rounded text-sm">
                    <i class="fas fa-redo mr-1"></i>Retry
                </button>
                ` : ''}
                <button onclick="deleteJob('${job.id}')" class="bg-red-500 hover:bg-red-700 text-white px-3 py-2 rounded text-sm">
                    <i class="fas fa-trash mr-1"></i>Delete
                </button>
            </div>
        </div>
    `).join('');
}

// Process Report functions
async function showProcessReport(jobId) {
    try {
        currentJobId = jobId;
        showSection('process-report');
        
        // Show loading state
        document.getElementById('process-report-content').innerHTML = `
            <div class="flex items-center justify-center py-8">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <span class="ml-3 text-gray-600 dark:text-gray-300">Loading process report...</span>
            </div>
        `;
        
        // Fetch comprehensive report
        const response = await fetch(`/jobs/${jobId}/comprehensive-report`);
        if (!response.ok) throw new Error('Failed to load comprehensive report');
        
        const reportData = await response.json();
        const report = reportData.report;
        
        // Display the comprehensive report
        displayProcessReport(report);
        
    } catch (error) {
        console.error('Error loading process report:', error);
        document.getElementById('process-report-content').innerHTML = `
            <div class="text-center py-8">
                <div class="text-red-500 mb-2">
                    <i class="fas fa-exclamation-triangle text-2xl"></i>
                </div>
                <p class="text-gray-600 dark:text-gray-300">Error loading process report: ${error.message}</p>
                <button onclick="showProcessReport('${jobId}')" class="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
                    <i class="fas fa-redo mr-2"></i>Retry
                </button>
            </div>
        `;
        showNotification('Error loading process report', 'error');
    }
}

function displayProcessReport(report) {
    const reportContent = document.getElementById('process-report-content');
    
    reportContent.innerHTML = `
        <!-- Executive Summary -->
        <div class="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-900 rounded-xl p-6 mb-6 border border-blue-200 dark:border-gray-700">
            <div class="flex items-center mb-4">
                <div class="bg-blue-500 text-white rounded-full p-3 mr-4">
                    <i class="fas fa-chart-line text-xl"></i>
                </div>
                <div>
                    <h3 class="text-xl font-bold text-gray-800 dark:text-white">Executive Summary</h3>
                    <p class="text-sm text-gray-600 dark:text-gray-400">Overall pipeline analysis</p>
                </div>
            </div>
            <p class="text-gray-700 dark:text-gray-300 leading-relaxed">${report.executive_summary}</p>
        </div>

        <!-- Key Metrics -->
        <div class="grid md:grid-cols-3 gap-6 mb-6">
            <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Efficiency Score</p>
                        <p class="text-2xl font-bold text-green-600 dark:text-green-400">${report.performance_metrics?.efficiency_score || 0}%</p>
                    </div>
                    <div class="bg-green-100 dark:bg-green-900 rounded-full p-3">
                        <i class="fas fa-tachometer-alt text-green-600 dark:text-green-400"></i>
                    </div>
                </div>
            </div>
            
            <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
                        <p class="text-2xl font-bold text-blue-600 dark:text-blue-400">${report.performance_metrics?.success_rate || 0}%</p>
                    </div>
                    <div class="bg-blue-100 dark:bg-blue-900 rounded-full p-3">
                        <i class="fas fa-check-circle text-blue-600 dark:text-blue-400"></i>
                    </div>
                </div>
            </div>
            
            <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Collaboration</p>
                        <p class="text-2xl font-bold text-purple-600 dark:text-purple-400">${report.performance_metrics?.collaboration_quality || 'N/A'}</p>
                    </div>
                    <div class="bg-purple-100 dark:bg-purple-900 rounded-full p-3">
                        <i class="fas fa-users text-purple-600 dark:text-purple-400"></i>
                    </div>
                </div>
            </div>
        </div>

        <!-- Pipeline Overview -->
        <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-6">
            <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                <i class="fas fa-cogs mr-2 text-blue-500"></i>Pipeline Overview
            </h3>
            <div class="grid md:grid-cols-2 gap-6">
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600 dark:text-gray-400">Learning Type:</span>
                        <span class="font-medium text-gray-800 dark:text-white capitalize">${report.pipeline_overview?.learning_type || 'Unknown'}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600 dark:text-gray-400">Total Execution Time:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${Math.round(report.pipeline_overview?.total_execution_time || 0)}s</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600 dark:text-gray-400">Models Generated:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${report.pipeline_overview?.models_generated || 0}</span>
                    </div>
                </div>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600 dark:text-gray-400">Agents Involved:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${report.pipeline_overview?.agents_involved || 0}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600 dark:text-gray-400">Messages Exchanged:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${report.pipeline_overview?.total_messages_exchanged || 0}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600 dark:text-gray-400">Tokens Consumed:</span>
                        <span class="font-medium text-gray-800 dark:text-white">${(report.pipeline_overview?.total_tokens_consumed || 0).toLocaleString()}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Agent Performance -->
        <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-6">
            <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                <i class="fas fa-robot mr-2 text-green-500"></i>Agent Performance Analysis
            </h3>
            <div class="grid gap-4">
                ${Object.entries(report.agent_analysis || {}).map(([agentName, analysis]) => `
                    <div class="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
                        <div class="flex items-center justify-between mb-3">
                            <h4 class="font-medium text-gray-800 dark:text-white">${agentName}</h4>
                            <span class="px-2 py-1 text-xs rounded-full ${getRatingBadgeClass(analysis.performance_rating)}">${analysis.performance_rating}</span>
                        </div>
                        <div class="grid md:grid-cols-3 gap-3 text-sm">
                            <div>
                                <span class="text-gray-600 dark:text-gray-400">Messages:</span>
                                <span class="font-medium text-gray-800 dark:text-white ml-1">${analysis.contributions?.total_messages || 0}</span>
                            </div>
                            <div>
                                <span class="text-gray-600 dark:text-gray-400">Outputs:</span>
                                <span class="font-medium text-gray-800 dark:text-white ml-1">${analysis.contributions?.key_outputs || 0}</span>
                            </div>
                            <div>
                                <span class="text-gray-600 dark:text-gray-400">Error Rate:</span>
                                <span class="font-medium text-gray-800 dark:text-white ml-1">${(analysis.efficiency?.error_rate * 100 || 0).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>

        <!-- Process Flow -->
        <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-6">
            <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                <i class="fas fa-stream mr-2 text-purple-500"></i>Process Flow
            </h3>
            <div class="space-y-3">
                ${(report.process_flow || []).map((step, index) => `
                    <div class="flex items-center p-3 border border-gray-200 dark:border-gray-600 rounded-lg">
                        <div class="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-4">
                            ${index + 1}
                        </div>
                        <div class="flex-grow">
                            <div class="flex items-center justify-between">
                                <h4 class="font-medium text-gray-800 dark:text-white">${step.agent}</h4>
                                <span class="text-xs px-2 py-1 rounded-full ${getQualityBadgeClass(step.output_quality)}">${step.output_quality}</span>
                            </div>
                            <p class="text-sm text-gray-600 dark:text-gray-400">${step.role}</p>
                            <p class="text-xs text-gray-500 dark:text-gray-500 mt-1">Duration: ${Math.round(step.duration || 0)}s</p>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>

        <!-- Key Achievements and Recommendations -->
        <div class="grid md:grid-cols-2 gap-6 mb-6">
            <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                    <i class="fas fa-trophy mr-2 text-yellow-500"></i>Key Achievements
                </h3>
                <ul class="space-y-2">
                    ${(report.key_achievements || []).map(achievement => `
                        <li class="flex items-start">
                            <i class="fas fa-check text-green-500 mr-2 mt-1 flex-shrink-0"></i>
                            <span class="text-gray-700 dark:text-gray-300">${achievement}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
            
            <div class="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                    <i class="fas fa-lightbulb mr-2 text-blue-500"></i>Recommendations
                </h3>
                <ul class="space-y-2">
                    ${(report.recommendations || []).map(recommendation => `
                        <li class="flex items-start">
                            <i class="fas fa-arrow-right text-blue-500 mr-2 mt-1 flex-shrink-0"></i>
                            <span class="text-gray-700 dark:text-gray-300">${recommendation}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        </div>
    `;
}

function getRatingBadgeClass(rating) {
    switch (rating) {
        case 'Excellent': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
        case 'Good': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300';
        case 'Fair': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
        default: return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
    }
}

function getQualityBadgeClass(quality) {
    switch (quality) {
        case 'High': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
        case 'Medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
        default: return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
    }
}

// Job details functions
async function loadJobDetails(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}`);
        const job = await response.json();
        
        // Update title
        document.getElementById('job-details-title').querySelector('span').textContent = job.name;
        
        // Update job info cards
        document.getElementById('job-details-info').innerHTML = `
            <div class="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-dark-700">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-gray-800 dark:text-white">Status</h3>
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusBadgeClass(job.status)}">
                        ${getStatusIcon(job.status)} ${job.status}
                    </span>
                </div>
                <div class="mb-3">
                    <div class="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                        <span>Progress</span>
                        <span>${job.progress}%</span>
                    </div>
                    <div class="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-2">
                        <div class="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500" style="width: ${job.progress}%"></div>
                    </div>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400">
                    Updated: ${formatDate(job.updated_at)}
                </div>
            </div>
            
            <div class="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-dark-700">
                <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4">Job Information</h3>
                <div class="space-y-3">
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Target Column</p>
                        <p class="text-sm text-gray-800 dark:text-white">${job.target_column}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Created</p>
                        <p class="text-sm text-gray-800 dark:text-white">${formatDate(job.created_at)}</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-dark-700">
                <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4">Objective</h3>
                <p class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">${job.prompt}</p>
                ${job.error_message ? `
                <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-sm font-medium text-red-800 dark:text-red-200 mb-2">Error</p>
                    <div class="max-h-32 overflow-y-auto bg-red-100/50 dark:bg-red-800/20 rounded p-2 border border-red-200/50 dark:border-red-700/30">
                        <pre class="text-xs text-red-700 dark:text-red-300 whitespace-pre-wrap font-mono leading-relaxed">${job.error_message}</pre>
                    </div>
                </div>` : ''}
            </div>
        `;
        
        // Show/hide retry button based on job status
        const retryBtn = document.getElementById('job-details-retry-btn');
        if (job.status === 'failed') {
            retryBtn.classList.remove('hidden');
        } else {
            retryBtn.classList.add('hidden');
        }
        
        // Load chat by default
        showJobTab('chat');
        loadJobChat(jobId, true); // Force scroll on first load
        startChatAutoRefresh(); // Start auto-refresh when job details are loaded
        
    } catch (error) {
        console.error('Error loading job details:', error);
        showNotification('Error loading job details', 'error');
    }
}

function closeJobModal() {
    document.getElementById('job-modal').classList.add('hidden');
    currentJobId = null;
    stopChatAutoRefresh();
}

function retryJobFromDetails() {
    if (currentJobId) {
        retryJob(currentJobId);
    }
}

// Job details tab functions
function showJobTab(tabName, event = null) {
    // Hide all tab content
    document.querySelectorAll('.job-tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // Show selected tab content
    document.getElementById('job-' + tabName + '-tab').classList.remove('hidden');
    
    // Update tab buttons
    document.querySelectorAll('.job-tab-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'text-blue-600', 'dark:text-blue-400');
        btn.classList.add('border-transparent', 'text-gray-500', 'dark:text-gray-400');
    });
    
    // Only update button styling if called from a click event
    if (event && event.target) {
        event.target.classList.remove('border-transparent', 'text-gray-500', 'dark:text-gray-400');
        event.target.classList.add('border-blue-500', 'text-blue-600', 'dark:text-blue-400');
    } else {
        // Find and activate the correct tab button
        const tabBtn = document.querySelector(`[onclick="showJobTab('${tabName}')"]`);
        if (tabBtn) {
            tabBtn.classList.remove('border-transparent', 'text-gray-500', 'dark:text-gray-400');
            tabBtn.classList.add('border-blue-500', 'text-blue-600', 'dark:text-blue-400');
        }
    }
    
    // Load tab-specific data
    if (tabName === 'chat') {
        loadJobChat(currentJobId);
    } else if (tabName === 'logs') {
        loadJobLogs(currentJobId);
    } else if (tabName === 'models') {
        loadJobModels(currentJobId);
    } else if (tabName === 'predictions') {
        loadJobPredictions(currentJobId);
    } else if (tabName === 'reports') {
        loadJobReports(currentJobId);
    } else if (tabName === 'statistics') {
        loadJobStatistics(currentJobId);
    } else if (tabName === 'scripts') {
        loadJobScripts(currentJobId);
    }
}

function showTab(tabName, event = null) {
    // Hide all tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // Show selected tab content
    document.getElementById(tabName + '-tab').classList.remove('hidden');
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'text-blue-600');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    
    // Only update button styling if called from a click event
    if (event && event.target) {
        event.target.classList.remove('border-transparent', 'text-gray-500');
        event.target.classList.add('border-blue-500', 'text-blue-600');
    }
    
    // Load tab-specific data
    if (tabName === 'chat') {
        loadJobChat(currentJobId);
    } else if (tabName === 'logs') {
        loadJobLogs(currentJobId);
    } else if (tabName === 'models') {
        loadJobModels(currentJobId);
    } else if (tabName === 'predictions') {
        loadJobPredictions(currentJobId);
    }
}

async function loadJobLogs(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/logs`);
        const logs = await response.json();
        
        const logsContent = document.getElementById('logs-content');
        
        if (logs.length === 0) {
            logsContent.innerHTML = '<p class="text-gray-400">No logs available</p>';
            return;
        }
        
        logsContent.innerHTML = logs.reverse().map(log => `
            <div class="mb-1">
                <span class="text-blue-400">${formatTime(log.timestamp)}</span>
                <span class="text-${log.level === 'ERROR' ? 'red' : 'green'}-400">[${log.level}]</span>
                <span>${log.message}</span>
            </div>
        `).join('');
        
        // Scroll to bottom
        logsContent.scrollTop = logsContent.scrollHeight;
        
    } catch (error) {
        console.error('Error loading logs:', error);
        document.getElementById('logs-content').innerHTML = '<p class="text-red-400">Error loading logs</p>';
    }
}

async function loadJobModels(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/models`);
        const models = await response.json();
        
        const modelsContent = document.getElementById('models-content');
        
        if (models.length === 0) {
            modelsContent.innerHTML = '<p class="text-gray-500 text-center">No models available</p>';
            return;
        }
        
        modelsContent.innerHTML = models.map(model => `
            <div class="border border-gray-200 rounded-lg p-4 mb-4">
                <div class="flex justify-between items-start">
                    <div>
                        <h4 class="font-semibold">${model.name}</h4>
                        <p class="text-gray-600 text-sm">Created: ${formatDate(model.created_at)}</p>
                        ${model.metrics ? `
                        <div class="mt-2">
                            <h5 class="text-sm font-medium text-gray-700">Metrics:</h5>
                            <div class="grid grid-cols-3 gap-2 mt-1">
                                ${Object.entries(model.metrics).map(([key, value]) => `
                                    <div class="text-sm">
                                        <span class="text-gray-600">${key}:</span>
                                        <span class="font-medium">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    <button onclick="downloadModel('${model.id}')" class="bg-blue-500 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm">
                        <i class="fas fa-download mr-1"></i>Download
                    </button>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('models-content').innerHTML = '<p class="text-red-500 text-center">Error loading models</p>';
    }
}

function loadJobPredictions(jobId) {
    // Placeholder for predictions
    document.getElementById('predictions-content').innerHTML = `
        <div class="text-center text-gray-500">
            <i class="fas fa-chart-line text-4xl mb-4"></i>
            <p>Predictions and visualizations will be displayed here</p>
            <p class="text-sm">Feature coming soon...</p>
        </div>
    `;
}

function refreshLogs() {
    if (currentJobId) {
        loadJobLogs(currentJobId);
    }
}

// Utility functions
async function updateModelMetrics(modelId, buttonElement = null) {
    let button = buttonElement;
    
    try {
        // Find button if not provided
        if (!button && event && event.target) {
            button = event.target.closest('button');
        }
        
        // Show loading state
        if (button) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i>Updating...';
        }
        
        const response = await fetch(`/models/${modelId}/update-metrics`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }
        
        const result = await response.json();
        
        // Show success notification
        showNotification(`Model metrics updated successfully! (${result.updated_metrics_count} metrics)`, 'success');
        
        // Reload the models to show updated metrics
        if (currentJobId) {
            loadJobModels(currentJobId);
        }
        
    } catch (error) {
        console.error('Error updating model metrics:', error);
        showNotification(`Error updating model metrics: ${error.message}`, 'error');
    } finally {
        // Reset button state
        if (button) {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-sync mr-2"></i>Update Metrics';
        }
    }
}

async function downloadModel(modelId) {
    try {
        window.open(`/models/${modelId}/download`, '_blank');
    } catch (error) {
        console.error('Error downloading model:', error);
        showNotification('Error downloading model', 'error');
    }
}

async function downloadResults(jobId) {
    try {
        // Get available results
        const response = await fetch(`/jobs/${jobId}/results`);
        const results = await response.json();
        
        if (results.predictions_csv) {
            window.open(`/jobs/${jobId}/predictions_csv`, '_blank');
            showNotification('Downloading predictions CSV', 'success');
        } else {
            showNotification('No prediction results available yet', 'info');
        }
    } catch (error) {
        console.error('Error downloading results:', error);
        showNotification('Error downloading results', 'error');
    }
}

async function retryJob(jobId) {
    if (!confirm('Are you sure you want to retry this job? This will restart the entire pipeline.')) {
        return;
    }
    
    try {
        const response = await fetch(`/jobs/${jobId}/retry`, {
            method: 'POST',
        });
        
        if (!response.ok) {
            throw new Error('Failed to retry job');
        }
        
        showNotification('Job retry initiated successfully', 'success');
        loadJobs();
        loadDashboard();
        
        // If job modal is open, refresh it
        if (currentJobId === jobId) {
            setTimeout(() => {
                openJobModal(jobId);
            }, 1000);
        }
               
    } catch (error) {
        console.error('Error retrying job:', error);
        showNotification('Error retrying job', 'error');
    }
}

async function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`/jobs/${jobId}`, {
            method: 'DELETE',
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete job');
        }
        
        showNotification('Job deleted successfully', 'success');
        loadJobs();
        loadDashboard();
        
    } catch (error) {
        console.error('Error deleting job:', error);
        showNotification('Error deleting job', 'error');
    }
}

// Job name editing functions
function editJobName(jobId) {
    console.log('Editing job name for ID:', jobId);
    // Hide display and show edit mode
    document.getElementById(`job-name-display-${jobId}`).classList.add('hidden');
    document.getElementById(`job-name-edit-${jobId}`).classList.remove('hidden');
    
    // Focus on the input
    const input = document.getElementById(`job-name-input-${jobId}`);
    input.focus();
    input.select();
    
    // Store original value
    input.dataset.originalValue = input.value;
    
    // Add enter key listener
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            saveJobName(jobId);
        } else if (e.key === 'Escape') {
            cancelEditJobName(jobId);
        }
    });
}

function cancelEditJobName(jobId) {
    const input = document.getElementById(`job-name-input-${jobId}`);
    
    // Restore original value
    input.value = input.dataset.originalValue || input.value;
    
    // Show display and hide edit mode
    document.getElementById(`job-name-display-${jobId}`).classList.remove('hidden');
    document.getElementById(`job-name-edit-${jobId}`).classList.add('hidden');
}

async function saveJobName(jobId) {
    const input = document.getElementById(`job-name-input-${jobId}`);
    const newName = input.value.trim();
    
    if (!newName) {
        showNotification('Job name cannot be empty', 'error');
        return;
    }
    
    if (newName === input.dataset.originalValue) {
        cancelEditJobName(jobId);
        return;
    }
    
    try {
        const response = await fetch(`/jobs/${jobId}/name`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: newName }),
        });
        
        if (!response.ok) {
            throw new Error('Failed to update job name');
        }
        
        // Update display
        document.getElementById(`job-name-display-${jobId}`).textContent = newName;
        
        // Show display and hide edit mode
        document.getElementById(`job-name-display-${jobId}`).classList.remove('hidden');
        document.getElementById(`job-name-edit-${jobId}`).classList.add('hidden');
        
        showNotification('Job name updated successfully', 'success');
        
    } catch (error) {
        console.error('Error updating job name:', error);
        showNotification('Error updating job name', 'error');
    }
}

function getStatusBadgeClass(status) {
    const classes = {
        'created': 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
        'processing': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
        'analyzing_data': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
        'training_model': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
        'generating_predictions': 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200',
        'awaiting_user_input': 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200 animate-pulse',
        'completed': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
        'failed': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
    };
    return classes[status] || 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
}

function getStatusIcon(status) {
    const icons = {
        'created': '<i class="fas fa-clock mr-1"></i>',
        'processing': '<i class="fas fa-spinner animate-spin mr-1"></i>',
        'analyzing_data': '<i class="fas fa-search mr-1"></i>',
        'training_model': '<i class="fas fa-brain mr-1"></i>',
        'generating_predictions': '<i class="fas fa-chart-line mr-1"></i>',
        'awaiting_user_input': '<i class="fas fa-user-edit mr-1 animate-pulse"></i>',
        'completed': '<i class="fas fa-check-circle mr-1"></i>',
        'failed': '<i class="fas fa-exclamation-triangle mr-1"></i>'
    };
    return icons[status] || '<i class="fas fa-question mr-1"></i>';
}

function getAgentColor(agentName) {
    const colors = {
        'Admin': 'bg-gradient-to-br from-purple-500 to-purple-600',
        'UserProxyAgent': 'bg-gradient-to-br from-purple-500 to-purple-600',
        'DataProcessorAgent': 'bg-gradient-to-br from-green-500 to-green-600',
        'ModelBuilderAgent': 'bg-gradient-to-br from-blue-500 to-blue-600',
        'CodeExecutorAgent': 'bg-gradient-to-br from-orange-500 to-orange-600',
        'AnalystAgent': 'bg-gradient-to-br from-red-500 to-red-600',
        'PredictionAgent': 'bg-gradient-to-br from-indigo-500 to-indigo-600',
        'VisualizationAgent': 'bg-gradient-to-br from-pink-500 to-pink-600',
        'Pipeline': 'bg-gradient-to-br from-gray-500 to-gray-600'
    };
    return colors[agentName] || 'bg-gradient-to-br from-gray-500 to-gray-600';
}

function getAgentBackgroundColor(agentName) {
    const colors = {
        'Admin': { 
            light: 'bg-purple-50 border-purple-200 dark:bg-purple-900/20 dark:border-purple-800/30',
            text: 'text-purple-800 dark:text-purple-200'
        },
        'UserProxyAgent': { 
            light: 'bg-purple-50 border-purple-200 dark:bg-purple-900/20 dark:border-purple-800/30',
            text: 'text-purple-800 dark:text-purple-200'
        },
        'DataProcessorAgent': { 
            light: 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800/30',
            text: 'text-green-800 dark:text-green-200'
        },
        'ModelBuilderAgent': { 
            light: 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800/30',
            text: 'text-blue-800 dark:text-blue-200'
        },
        'CodeExecutorAgent': { 
            light: 'bg-orange-50 border-orange-200 dark:bg-orange-900/20 dark:border-orange-800/30',
            text: 'text-orange-800 dark:text-orange-200'
        },
        'AnalystAgent': { 
            light: 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800/30',
            text: 'text-red-800 dark:text-red-200'
        },
        'PredictionAgent': { 
            light: 'bg-indigo-50 border-indigo-200 dark:bg-indigo-900/20 dark:border-indigo-800/30',
            text: 'text-indigo-800 dark:text-indigo-200'
        },
        'VisualizationAgent': { 
            light: 'bg-pink-50 border-pink-200 dark:bg-pink-900/20 dark:border-pink-800/30',
            text: 'text-pink-800 dark:text-pink-200'
        },
        'Pipeline': { 
            light: 'bg-gray-50 border-gray-200 dark:bg-gray-800/50 dark:border-gray-700/50',
            text: 'text-gray-800 dark:text-gray-200'
        }
    };
    return colors[agentName] || colors['Pipeline'];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString();
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'success' ? 'bg-green-500 text-white' :
        type === 'error' ? 'bg-red-500 text-white' :
        type === 'warning' ? 'bg-yellow-500 text-white' :
        'bg-blue-500 text-white'
    }`;
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-${
                type === 'success' ? 'check-circle' :
                type === 'error' ? 'exclamation-circle' :
                type === 'warning' ? 'exclamation-triangle' :
                'info-circle'
            } mr-2"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Chat functions
async function loadJobChat(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/messages`);
        const messages = await response.json();
        
        const chatContainer = document.getElementById('chat-messages');
        const userInputArea = document.getElementById('user-input-area');
        
        if (filteredMessages.length === 0) {
            chatContainer.innerHTML = '<div class="text-center text-gray-500 py-8">No agent messages yet. The agents will start communicating once the job begins.</div>';
            if (userInputArea) {
                userInputArea.classList.add('hidden');
            }
            return;
        }
        
        // Filter out unwanted messages
        const filteredMessages = messages.filter(message => {
            const agentName = message.agent_name || '';
            const content = message.content || '';
            const unwantedTerms = ['Pipeline', 'AgentEvent', 'ScriptCapture', 'Event'];
            
            // Check if agent name or content contains any unwanted terms
            return !unwantedTerms.some(term => 
                agentName.toLowerCase().includes(term.toLowerCase()) ||
                content.toLowerCase().includes(term.toLowerCase())
            );
        });

        // Render messages
        chatContainer.innerHTML = filteredMessages.map(message => {
            if (message.source === 'user') {
                return `
                    <div class="flex justify-end animate-slide-up">
                        <div class="bg-gradient-to-br from-blue-500 to-blue-600 text-white px-4 py-3 rounded-2xl max-w-xs lg:max-w-md shadow-lg">
                            <div class="flex items-center mb-1">
                                <div class="w-6 h-6 bg-white/20 rounded-full flex items-center justify-center mr-2">
                                    <i class="fas fa-user text-xs"></i>
                                </div>
                                <div class="text-sm font-semibold">You</div>
                            </div>
                            <div class="text-sm leading-relaxed">${escapeHtml(message.content)}</div>
                            <div class="text-xs text-blue-100 mt-2 opacity-75">${formatTime(message.timestamp)}</div>
                        </div>
                    </div>
                `;
            } else {
                const agentColor = getAgentColor(message.agent_name);
                const bgColor = getAgentBackgroundColor(message.agent_name);
                const isUserInputRequest = message.message_type === 'user_input_request';
                const specialClasses = isUserInputRequest ? 'border-4 border-orange-300 dark:border-orange-700 bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/30 dark:to-yellow-900/30 animate-pulse' : bgColor.light;
                
                return `
                    <div class="flex justify-start animate-slide-up">
                        <div class="${specialClasses} border px-4 py-3 rounded-2xl max-w-xs lg:max-w-md shadow-lg">
                            <div class="flex items-center mb-2">
                                <div class="w-8 h-8 ${agentColor} rounded-full flex items-center justify-center mr-3 shadow-md">
                                    <i class="fas ${isUserInputRequest ? 'fa-user-edit animate-pulse' : 'fa-robot'} text-white text-sm"></i>
                                </div>
                                <div class="text-sm font-semibold ${bgColor.text}">${escapeHtml(message.agent_name)}</div>
                            </div>
                            <div class="text-sm ${bgColor.text} leading-relaxed">${formatMessageContent(message.content)}</div>
                            <div class="text-xs mt-2 opacity-75 ${bgColor.text}">${formatTime(message.timestamp)}</div>
                        </div>
                    </div>
                `;
            }
        }).join('');
        
        // Check if we need to show user input (job status indicates agents are waiting)
        const jobResponse = await fetch(`/jobs/${jobId}`);
        const job = await jobResponse.json();
        
        // Check if any message is requesting user input
        const hasUserInputRequest = filteredMessages.some(msg => 
            msg.message_type === 'user_input_request' || 
            msg.content.toLowerCase().includes('enter your response:') ||
            msg.content.toLowerCase().includes('user input requested')
        );
        
        // Always show user input when there are messages, but highlight if input is requested
        if (filteredMessages.length > 0) {
            userInputArea.classList.remove('hidden');
            
            // Add visual indicator if user input is requested
            if (hasUserInputRequest || job.status === 'awaiting_user_input') {
                userInputArea.classList.add('bg-yellow-50', 'dark:bg-yellow-900/20', 'border-yellow-300', 'dark:border-yellow-700');
                
                // Show notification if input is needed
                if (hasUserInputRequest && !userInputArea.dataset.notificationShown) {
                    showNotification('🔔 Agent is waiting for your input!', 'warning');
                    userInputArea.dataset.notificationShown = 'true';
                }
            } else {
                userInputArea.classList.remove('bg-yellow-50', 'dark:bg-yellow-900/20', 'border-yellow-300', 'dark:border-yellow-700');
                userInputArea.dataset.notificationShown = 'false';
            }
        } else {
            userInputArea.classList.add('hidden');
        }
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
    } catch (error) {
        console.error('Error loading chat messages:', error);
        document.getElementById('chat-messages').innerHTML = '<div class="text-center text-red-500 py-8">Error loading chat messages</div>';
    }
}

async function sendUserMessage() {
    const input = document.getElementById('user-message-input');
    const message = input.value.trim();
    
    if (!message || !currentJobId) {
        return;
    }
    
    // Disable input and button while sending
    const button = event.target;
    const originalButtonText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i>Sending...';
    input.disabled = true;
    
    try {
        const response = await fetch(`/jobs/${currentJobId}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send message');
        }
        
        input.value = '';
        
        // Reload chat immediately to show the sent message
        await loadJobChat(currentJobId, true);

        showNotification('Message sent to agents', 'success');
        
    } catch (error) {
        console.error('Error sending message:', error);
        showNotification('Error sending message', 'error');
    } finally {
        // Re-enable input and button
        button.disabled = false;
        button.innerHTML = originalButtonText;
        input.disabled = false;
        input.focus();
    }
}

function refreshChat() {
    if (currentJobId) {
        loadJobChat(currentJobId);
    }
}

function formatMessageContent(content) {
    // Configure marked for better rendering
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.warn('Highlight.js error:', err);
                    }
                }
                return hljs ? hljs.highlightAuto(code).value : escapeHtml(code);
            },
            breaks: true,
            gfm: true
        });
        
        try {
            // Use marked to parse markdown
            let htmlContent = marked.parse(content);
            
            // Add custom classes for dark mode
            htmlContent = htmlContent.replace(/<pre><code/g, '<pre class="hljs bg-gray-900 dark:bg-gray-800 text-white rounded-lg p-4 my-3 overflow-x-auto text-sm"><code');
            htmlContent = htmlContent.replace(/<code(?![^>]*class)/g, '<code class="bg-gray-200 dark:bg-gray-700 text-red-600 dark:text-red-400 px-1 py-0.5 rounded text-xs"');
            htmlContent = htmlContent.replace(/<p>/g, '<p class="mb-2 last:mb-0">');
            htmlContent = htmlContent.replace(/<ul>/g, '<ul class="list-disc list-inside mb-2">');
            htmlContent = htmlContent.replace(/<ol>/g, '<ol class="list-decimal list-inside mb-2">');
            htmlContent = htmlContent.replace(/<li>/g, '<li class="mb-1">');
            htmlContent = htmlContent.replace(/<blockquote>/g, '<blockquote class="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic text-gray-700 dark:text-gray-300">');
            
            return htmlContent;
        } catch (error) {
            console.warn('Markdown parsing error:', error);
        }
    }
    
    // Fallback for basic formatting
    let formattedContent = escapeHtml(content);
    
    // Handle code blocks with syntax highlighting
    if (formattedContent.includes('```')) {
        formattedContent = formattedContent.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
            const unescapedCode = code.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
            let highlightedCode = unescapedCode;
            
            if (typeof hljs !== 'undefined') {
                try {
                    if (lang && hljs.getLanguage(lang)) {
                        highlightedCode = hljs.highlight(unescapedCode, { language: lang }).value;
                    } else {
                        highlightedCode = hljs.highlightAuto(unescapedCode).value;
                    }
                } catch (err) {
                    console.warn('Highlight.js error:', err);
                    highlightedCode = escapeHtml(unescapedCode);
                }
            }
            
            return `<pre class="hljs bg-gray-900 dark:bg-gray-800 text-white rounded-lg p-4 my-3 overflow-x-auto text-sm"><code>${highlightedCode}</code></pre>`;
        });
    }
    
    // Handle inline code
    formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code class="bg-gray-200 dark:bg-gray-700 text-red-600 dark:text-red-400 px-1 py-0.5 rounded text-xs">$1</code>');
    
    // Convert line breaks to HTML
    formattedContent = formattedContent.replace(/\n/g, '<br>');
    
    return formattedContent;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-refresh functionality
let chatRefreshInterval = null;

function startChatAutoRefresh() {
    // Stop any existing refresh
    stopChatAutoRefresh();
    
    // Start new refresh every 3 seconds
    chatRefreshInterval = setInterval(() => {
        if (currentJobId && document.getElementById('job-chat-tab') && !document.getElementById('job-chat-tab').classList.contains('hidden')) {
            loadJobChat(currentJobId);
        }
    }, 3000);
}

function stopChatAutoRefresh() {
    if (chatRefreshInterval) {
        clearInterval(chatRefreshInterval);
        chatRefreshInterval = null;
    }
}

// Job details specific chat functions
let isUserScrolling = false;
let lastScrollTop = 0;
let lastMessageCount = 0;

async function loadJobChat(jobId, forceScrollDown = false) {
    try {
        const response = await fetch(`/jobs/${jobId}/messages`);
        const messages = await response.json();
        
        const chatContainer = document.getElementById('job-chat-messages');
        const userInputArea = document.getElementById('job-user-input-area');
        const scrollButton = document.getElementById('scroll-to-bottom');
        
        // Verificar que los elementos existan antes de continuar
        if (!chatContainer) {
            console.warn('Chat container not found');
            return;
        }

        // Filter out unwanted messages first
        const filteredMessages = messages.filter(message => {
            const agentName = message.agent_name || '';
            const content = message.content || '';
            const unwantedTerms = ['Pipeline', 'AgentEvent', 'ScriptCapture', 'Event'];
            
            // Check if agent name or content contains any unwanted terms
            return !unwantedTerms.some(term => 
                agentName.toLowerCase().includes(term.toLowerCase()) ||
                content.toLowerCase().includes(term.toLowerCase())
            );
        });

        // Si no hay mensajes nuevos, no hagas nada para evitar el "salto" visual.
        if (filteredMessages.length === lastMessageCount && !forceScrollDown) {
            return;
        }

        // 1. Comprueba si el usuario está al final del chat ANTES de añadir nuevos mensajes.
        // El margen de 50px da un poco de flexibilidad.
        const isAtBottom = chatContainer.scrollTop + chatContainer.clientHeight >= chatContainer.scrollHeight - 50;
        const hasNewMessages = filteredMessages.length > lastMessageCount;

        if (filteredMessages.length === 0) {
            chatContainer.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-comments text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No agent messages yet</p>
                    <p class="text-sm text-gray-400 dark:text-gray-500">Agents will start communicating once the job begins</p>
                </div>
            `;
            if (userInputArea) {
                userInputArea.classList.add('hidden');
            }
            return;
        }
        
        lastMessageCount = filteredMessages.length;
        
        // Renderiza los mensajes
        chatContainer.innerHTML = filteredMessages.map(message => {
            if (message.source === 'user') {
                return `
                    <div class="flex justify-end animate-slide-up">
                        <div class="bg-gradient-to-br from-blue-500 to-blue-600 text-white px-6 py-4 rounded-2xl shadow-lg">
                            <div class="flex items-center mb-2">
                                <div class="w-7 h-7 bg-white/20 rounded-full flex items-center justify-center mr-3">
                                    <i class="fas fa-user text-sm"></i>
                                </div>
                                <div class="text-base font-semibold">You</div>
                            </div>
                            <div class="text-base leading-relaxed whitespace-pre-wrap">${escapeHtml(message.content)}</div>
                            <div class="text-xs text-blue-100 mt-3 opacity-75">${formatTime(message.timestamp)}</div>
                        </div>
                    </div>
                `;
            } else {
                const agentColor = getAgentColor(message.agent_name);
                const bgColor = getAgentBackgroundColor(message.agent_name);
                const isUserInputRequest = message.message_type === 'user_input_request';
                const specialClasses = isUserInputRequest ? 'border-4 border-orange-300 dark:border-orange-700 bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/30 dark:to-yellow-900/30 animate-pulse' : bgColor.light;
                
                return `
                    <div class="flex justify-start animate-slide-up">
                        <div class="${specialClasses} border px-6 py-4 rounded-2xl shadow-lg">
                            <div class="flex items-center mb-3">
                                <div class="w-9 h-9 ${agentColor} rounded-full flex items-center justify-center mr-3 shadow-md">
                                    <i class="fas ${isUserInputRequest ? 'fa-user-edit animate-pulse' : 'fa-robot'} text-white"></i>
                                </div>
                                <div class="text-base font-semibold ${bgColor.text}">${escapeHtml(message.agent_name)}</div>
                            </div>
                            <div class="prose prose-base max-w-none ${bgColor.text} leading-relaxed">${formatMessageContent(message.content)}</div>
                            <div class="text-xs mt-3 opacity-75 ${bgColor.text}">${formatTime(message.timestamp)}</div>
                        </div>
                    </div>
                `;
            }
        }).join('');        
       
        const jobResponse = await fetch(`/jobs/${jobId}`);
        const job = await jobResponse.json();
        
        const hasUserInputRequest = filteredMessages.some(msg => 
            msg.message_type === 'user_input_request' || 
            msg.content.toLowerCase().includes('enter your response:') ||
            msg.content.toLowerCase().includes('user input requested')
        );
        
        if (filteredMessages.length > 0 && userInputArea) {
            userInputArea.classList.remove('hidden');
            if (hasUserInputRequest || job.status === 'awaiting_user_input') {
                userInputArea.classList.add('bg-yellow-50', 'dark:bg-yellow-900/20', 'border-yellow-300', 'dark:border-yellow-700');
                if (hasUserInputRequest && !userInputArea.dataset.notificationShown) {
                    showNotification('🔔 Agent is waiting for your input!', 'warning');
                    userInputArea.dataset.notificationShown = 'true';
                    setTimeout(() => {
                        userInputArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 500);
                }
            } else {
                userInputArea.classList.remove('bg-yellow-50', 'dark:bg-yellow-900/20', 'border-yellow-300', 'dark:border-yellow-700');
                userInputArea.dataset.notificationShown = 'false';
            }
        } else if (userInputArea) {
            userInputArea.classList.add('hidden');
        }
        
        // 2. Decide whether to scroll or show the new messages button.
        if (isAtBottom || forceScrollDown) {
            // Si el usuario estaba al final, llévalo al nuevo final.
            scrollChatToBottom();
        } else if (hasNewMessages && scrollButton) {
            // Si el usuario estaba arriba y hay nuevos mensajes, muestra el botón.
            scrollButton.classList.remove('opacity-0', 'invisible');
        }
        
    } catch (error) {
        console.error('Error loading chat messages:', error);
        document.getElementById('job-chat-messages').innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <p class="text-red-500">Error loading chat messages</p>
            </div>
        `;
    }
}

async function sendJobUserMessage() {
    const input = document.getElementById('job-user-message-input');
    const message = input.value.trim();
    
    if (!message || !currentJobId) {
        return;
    }
    
    // Disable input and button while sending
    const button = event.target;
    const originalButtonText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i>Sending...';
    input.disabled = true;
    
    try {
        const response = await fetch(`/jobs/${currentJobId}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send message');
        }
        
        input.value = '';
        
        // Reload chat immediately to show the sent message
        await loadJobChat(currentJobId, true);
        
        showNotification('Message sent to agents', 'success');
        
    } catch (error) {
        console.error('Error sending message:', error);
        showNotification('Error sending message', 'error');
    } finally {
        // Re-enable input and button
        button.disabled = false;
        button.innerHTML = originalButtonText;
        input.disabled = false;
        input.focus();
    }
}

function refreshJobChat() {
    if (currentJobId) {
        loadJobChat(currentJobId);
    }
}

async function loadJobLogs(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/logs`);
        const logs = await response.json();
        
        const logsContent = document.getElementById('job-logs-content');
        
        if (logs.length === 0) {
            logsContent.innerHTML = '<p class="text-gray-400">No logs available</p>';
            return;
        }
        
        logsContent.innerHTML = logs.reverse().map(log => `
            <div class="mb-1">
                <span class="text-blue-400">${formatTime(log.timestamp)}</span>
                <span class="text-${log.level === 'ERROR' ? 'red' : 'green'}-400">[${log.level}]</span>
                <span>${log.message}</span>
            </div>
        `).join('');
        
        // Scroll to bottom
        logsContent.scrollTop = logsContent.scrollHeight;
        
    } catch (error) {
        console.error('Error loading logs:', error);
        document.getElementById('job-logs-content').innerHTML = '<p class="text-red-400">Error loading logs</p>';
    }
}

async function loadJobModels(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/models`);
        const models = await response.json();
        
        const modelsContent = document.getElementById('job-models-content');
        
        if (models.length === 0) {
            modelsContent.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-brain text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No models generated yet</p>
                </div>
            `;
            return;
        }
        
        modelsContent.innerHTML = models.map(model => `
            <div class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 mb-6">
                <div class="flex justify-between items-start mb-4">
                    <div class="flex-1">
                        <h4 class="text-xl font-bold text-gray-900 dark:text-white mb-2 flex items-center">
                            <i class="fas fa-brain mr-3 text-blue-500"></i>
                            ${escapeHtml(model.name)}
                        </h4>
                        <div class="flex items-center text-sm text-gray-600 dark:text-gray-300 mb-3">
                            <i class="fas fa-calendar mr-2 text-gray-400"></i>
                            <span>Created: ${formatTime(model.created_at)}</span>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="updateModelMetrics('${model.id}')" class="bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white px-3 py-2 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg transform hover:scale-105 text-sm">
                            <i class="fas fa-sync mr-2"></i>Update Metrics
                        </button>
                        <button onclick="downloadModel('${model.id}')" class="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg transform hover:scale-105">
                            <i class="fas fa-download mr-2"></i>Download
                        </button>
                    </div>
                </div>
                
                ${model.metrics ? createModelMetricsDisplay(model.metrics) : ''}
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('job-models-content').innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <p class="text-red-500">Error loading models</p>
            </div>
        `;
    }
}

async function loadJobPredictions(jobId) {
    try {
        // Get job results info
        const response = await fetch(`/jobs/${jobId}/results`);
        const results = await response.json();
        
        const predictionsContent = document.getElementById('job-predictions-content');
        
        if (!results.forecast_plot && !results.predictions_csv) {
            predictionsContent.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-chart-line text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No predictions generated yet</p>
                    <p class="text-sm text-gray-400 dark:text-gray-500">Predictions and visualizations will appear here once the job completes</p>
                </div>
            `;
            return;
        }
        
        let content = '<div class="space-y-6">';
        
        // Show forecast plot if available
        if (results.forecast_plot) {
            content += `
                <div class="bg-white dark:bg-dark-700 border border-gray-200 dark:border-dark-600 rounded-lg p-6">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-semibold text-gray-800 dark:text-white">
                            <i class="fas fa-chart-line mr-2 text-blue-500"></i>Forecast Visualization
                        </h3>
                        <button onclick="downloadForecastPlot('${jobId}')" class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-2 rounded text-sm transition-colors">
                            <i class="fas fa-download mr-1"></i>Download
                        </button>
                    </div>
                    <div class="bg-gray-50 dark:bg-dark-800 rounded-lg p-4">
                        <img src="/jobs/${jobId}/forecast_plot" 
                             alt="Forecast Plot" 
                             class="w-full h-auto rounded-lg shadow-lg" 
                             onerror="this.onerror=null; this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5YTNhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBhdmFpbGFibGU8L3RleHQ+PC9zdmc+'" />
                    </div>
                </div>
            `;
        }
        
        // Show predictions CSV download if available
        if (results.predictions_csv) {
            content += `
                <div class="bg-white dark:bg-dark-700 border border-gray-200 dark:border-dark-600 rounded-lg p-6">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-semibold text-gray-800 dark:text-white">
                            <i class="fas fa-table mr-2 text-green-500"></i>Predictions Data
                        </h3>
                        <button onclick="downloadPredictionsCsv('${jobId}')" class="bg-green-500 hover:bg-green-600 text-white px-3 py-2 rounded text-sm transition-colors">
                            <i class="fas fa-download mr-1"></i>Download CSV
                        </button>
                    </div>
                    <div class="bg-gray-50 dark:bg-dark-800 rounded-lg p-4">
                        <p class="text-gray-600 dark:text-gray-400">
                            <i class="fas fa-file-csv mr-2"></i>
                            Prediction results are available for download as CSV file
                        </p>
                    </div>
                </div>
            `;
        }
        
        content += '</div>';
        predictionsContent.innerHTML = content;
        
    } catch (error) {
        console.error('Error loading predictions:', error);
        document.getElementById('job-predictions-content').innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <p class="text-red-500">Error loading predictions</p>
            </div>
        `;
    }
}

function refreshJobLogs() {
    if (currentJobId) {
        loadJobLogs(currentJobId);
    }
}

function createModelMetricsDisplay(metrics) {
    if (!metrics) return '';
    
    // Categorizar métricas para una mejor organización
    const basicInfo = {};
    const modelSpecificMetrics = {};
    const performanceMLMetrics = {};
    const hyperparameters = {};
    const trainingMetrics = {};
    const performanceMetrics = {};
    const processMetrics = {};
    const agentMetrics = {};
    
    Object.entries(metrics).forEach(([key, value]) => {
        if (['model_name', 'model_type', 'model_algorithm', 'model_family', 'model_description', 'target_column', 'file_size_mb', 'creation_timestamp', 'is_automl_leader'].includes(key)) {
            basicInfo[key] = value;
        } else if (key.includes('estimated_') || ['cross_validation', 'feature_importance', 'partial_dependence', 'shap_values'].includes(key)) {
            performanceMLMetrics[key] = value;
        } else if (key.startsWith('hp_') || key.includes('automl_')) {
            hyperparameters[key] = value;
        } else if (key.includes('training') || key.includes('duration') || key.includes('time')) {
            trainingMetrics[key] = value;
        } else if (['efficiency_score', 'messages_per_minute', 'average_message_interval'].includes(key)) {
            performanceMetrics[key] = value;
        } else if (['framework_used', 'pipeline_version', 'multi_agent_system', 'docker_execution'].includes(key)) {
            processMetrics[key] = value;
        } else if (key.includes('agent') || key.includes('participation')) {
            agentMetrics[key] = value;
        } else {
            modelSpecificMetrics[key] = value; // Otros datos específicos del modelo
        }
    });
    
    let html = `
        <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h5 class="font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <i class="fas fa-chart-bar mr-2 text-green-500"></i>
                Métricas del Modelo
            </h5>
    `;
    
    // Basic Information
    if (Object.keys(basicInfo).length > 0) {
        html += `
            <div class="mb-6">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-info-circle mr-2 text-blue-500"></i>
                    Model Information
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(basicInfo)}
                </div>
            </div>
        `;
    }
    
    // ML Performance Metrics
    if (Object.keys(performanceMLMetrics).length > 0) {
        html += `
            <div class="mb-6">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-chart-line mr-2 text-green-500"></i>
                    Machine Learning Metrics
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(performanceMLMetrics)}
                </div>
            </div>
        `;
    }
    
    // Hiperparámetros
    if (Object.keys(hyperparameters).length > 0) {
        html += `
            <div class="mb-6">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-sliders-h mr-2 text-indigo-500"></i>
                    Hiperparámetros
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(hyperparameters)}
                </div>
            </div>
        `;
    }
    
    // Métricas de Entrenamiento
    if (Object.keys(trainingMetrics).length > 0) {
        html += `
            <div class="mb-6">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-cogs mr-2 text-purple-500"></i>
                    Entrenamiento
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(trainingMetrics)}
                </div>
            </div>
        `;
    }
    
    // Métricas de Rendimiento
    if (Object.keys(performanceMetrics).length > 0) {
        html += `
            <div class="mb-6">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-tachometer-alt mr-2 text-orange-500"></i>
                    Rendimiento
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(performanceMetrics)}
                </div>
            </div>
        `;
    }
    
    // Métricas de Agentes
    if (Object.keys(agentMetrics).length > 0) {
        html += `
            <div class="mb-6">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-users mr-2 text-green-500"></i>
                    Sistema Multi-Agente
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(agentMetrics)}
                </div>
            </div>
        `;
    }
    
    // Métricas del Proceso
    if (Object.keys(processMetrics).length > 0) {
        html += `
            <div class="mb-4">
                <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <i class="fas fa-microchip mr-2 text-red-500"></i>
                    System Configuration
                </h6>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    ${createMetricCards(processMetrics)}
                </div>
            </div>
        `;
    }
    
    html += `</div>`;
    return html;
}

function createMetricCards(metricsObj) {
    return Object.entries(metricsObj).map(([key, value]) => {
        let displayValue = value;
        let valueClass = 'text-gray-900 dark:text-white';
        let icon = 'fas fa-info';
        
        // Formatear valores especiales
        if (typeof value === 'number') {
            if (key.includes('score') || key.includes('efficiency')) {
                displayValue = `${value}%`;
                if (value >= 80) {
                    valueClass = 'text-green-600 dark:text-green-400';
                    icon = 'fas fa-check-circle';
                } else if (value >= 60) {
                    valueClass = 'text-yellow-600 dark:text-yellow-400';
                    icon = 'fas fa-exclamation-circle';
                } else {
                    valueClass = 'text-red-600 dark:text-red-400';
                    icon = 'fas fa-times-circle';
                }
            } else if (key.includes('accuracy') || key.includes('auc')) {
                displayValue = `${(value * 100).toFixed(1)}%`;
                if (value >= 0.9) {
                    valueClass = 'text-green-600 dark:text-green-400';
                    icon = 'fas fa-trophy';
                } else if (value >= 0.8) {
                    valueClass = 'text-blue-600 dark:text-blue-400';
                    icon = 'fas fa-medal';
                } else if (value >= 0.7) {
                    valueClass = 'text-yellow-600 dark:text-yellow-400';
                    icon = 'fas fa-star';
                } else {
                    valueClass = 'text-orange-600 dark:text-orange-400';
                    icon = 'fas fa-chart-line';
                }
            } else if (key.includes('logloss') || key.includes('error')) {
                displayValue = value.toFixed(4);
                if (value <= 0.2) {
                    valueClass = 'text-green-600 dark:text-green-400';
                    icon = 'fas fa-arrow-down';
                } else if (value <= 0.4) {
                    valueClass = 'text-yellow-600 dark:text-yellow-400';
                    icon = 'fas fa-minus';
                } else {
                    valueClass = 'text-red-600 dark:text-red-400';
                    icon = 'fas fa-arrow-up';
                }
            } else if (key.includes('mb') || key.includes('size')) {
                displayValue = `${value} MB`;
                icon = 'fas fa-hdd';
            } else if (key.includes('time') || key.includes('duration')) {
                if (key.includes('minutes')) {
                    displayValue = `${value} min`;
                } else {
                    displayValue = `${value} sec`;
                }
                icon = 'fas fa-clock';
            } else if (key.includes('messages') || key.includes('count')) {
                displayValue = value.toLocaleString();
                icon = 'fas fa-comments';
            } else {
                displayValue = value.toFixed(2);
            }
        } else if (typeof value === 'boolean') {
            displayValue = value ? 'Sí' : 'No';
            valueClass = value ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
            icon = value ? 'fas fa-check' : 'fas fa-times';
        } else if (typeof value === 'object') {
            displayValue = JSON.stringify(value, null, 2);
            icon = 'fas fa-code';
        } else if (key.includes('timestamp')) {
            displayValue = new Date(value).toLocaleString();
            icon = 'fas fa-calendar';
        }
        
        // Mejorar el nombre de la métrica
        let displayName = key.replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace(/Mb/g, 'MB')
            .replace(/Id/g, 'ID')
            .replace(/Hp /g, '')  // Remover prefijo HP de hiperparámetros
            .replace(/Estimated /g, '')  // Remover prefijo Estimated
            .replace(/Automl/g, 'AutoML');
            
        // Iconos específicos para métricas de ML
        if (key.startsWith('hp_')) {
            icon = 'fas fa-sliders-h';
        } else if (key.includes('accuracy')) {
            icon = 'fas fa-bullseye';
        } else if (key.includes('auc')) {
            icon = 'fas fa-chart-area';
        } else if (key.includes('algorithm')) {
            icon = 'fas fa-brain';
        } else if (key.includes('family')) {
            icon = 'fas fa-sitemap';
        } else if (key.includes('leader')) {
            icon = value ? 'fas fa-crown' : 'fas fa-medal';
            if (value) {
                valueClass = 'text-yellow-600 dark:text-yellow-400';
            }
        } else if (key.includes('feature_importance')) {
            icon = 'fas fa-list-ol';
        } else if (key.includes('cross_validation')) {
            icon = 'fas fa-sync-alt';
        } else if (key.includes('shap')) {
            icon = 'fas fa-search-plus';
        }
            
        return `
            <div class="bg-white dark:bg-gray-900 p-3 rounded-lg border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow">
                <div class="flex items-center mb-1">
                    <i class="${icon} mr-2 text-xs text-gray-500 dark:text-gray-400"></i>
                    <div class="font-medium text-gray-600 dark:text-gray-300 text-xs uppercase tracking-wide">
                        ${displayName}
                    </div>
                </div>
                <div class="${valueClass} font-semibold text-sm break-words">
                    ${typeof value === 'object' ? `<pre class="text-xs overflow-hidden">${displayValue}</pre>` : displayValue}
                </div>
            </div>
        `;
    }).join('');
}

function scrollChatToBottom() {
    const chatContainer = document.getElementById('job-chat-messages');
    if (chatContainer) {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
        
        // Oculta el botón de "scroll to bottom" después de hacer clic o al llegar al final.
        const scrollButton = document.getElementById('scroll-to-bottom');
        if (scrollButton) {
            scrollButton.classList.add('opacity-0', 'invisible');
        }
    }
}

// New download functions for predictions
async function downloadForecastPlot(jobId) {
    try {
        window.open(`/jobs/${jobId}/forecast_plot`, '_blank');
        showNotification('Downloading forecast plot', 'success');
    } catch (error) {
        console.error('Error downloading forecast plot:', error);
        showNotification('Error downloading forecast plot', 'error');
    }
}

async function downloadPredictionsCsv(jobId) {
    try {
        window.open(`/jobs/${jobId}/predictions_csv`, '_blank');
        showNotification('Downloading predictions CSV', 'success');
    } catch (error) {
        console.error('Error downloading predictions CSV:', error);
        showNotification('Error downloading predictions CSV', 'error');
    }
}

// Process Reports Functions
async function loadJobReports(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/reports`);
        const reports = await response.json();
        
        const reportsContent = document.getElementById('job-reports-content');
        
        if (reports.length === 0) {
            reportsContent.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-file-invoice text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No process reports available yet</p>
                    <p class="text-sm text-gray-400 dark:text-gray-500">Reports will be generated as the pipeline progresses</p>
                </div>
            `;
            return;
        }
        
        // Group reports by stage
        const reportsByStage = {};
        reports.forEach(report => {
            if (!reportsByStage[report.stage]) {
                reportsByStage[report.stage] = [];
            }
            reportsByStage[report.stage].push(report);
        });
        
        const stageOrder = ['dataset', 'preprocessing', 'training', 'model', 'predictions', 'summary'];
        let content = '';
        
        stageOrder.forEach(stage => {
            if (reportsByStage[stage]) {
                const stageReports = reportsByStage[stage];
                const stageColors = {
                    'dataset': 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
                    'preprocessing': 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
                    'training': 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
                    'model': 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
                    'predictions': 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800',
                    'summary': 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800'
                };
                
                const stageIcons = {
                    'dataset': 'fa-database',
                    'preprocessing': 'fa-cogs',
                    'training': 'fa-brain',
                    'model': 'fa-cube',
                    'predictions': 'fa-chart-line',
                    'summary': 'fa-flag-checkered'
                };
                
                const stageColor = stageColors[stage] || 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
                const stageIcon = stageIcons[stage] || 'fa-file';
                
                content += `
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4 flex items-center">
                            <i class="fas ${stageIcon} mr-2 text-blue-500"></i>
                            ${stage.charAt(0).toUpperCase() + stage.slice(1)} Reports
                        </h3>
                        <div class="space-y-3">
                `;
                
                stageReports.forEach(report => {
                    content += `
                        <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-6 ${stageColor}">
                            <div class="flex justify-between items-start mb-3">
                                <h4 class="font-semibold text-gray-800 dark:text-white">${report.title}</h4>
                                <span class="text-xs text-gray-500 dark:text-gray-400">${formatDate(report.created_at)}</span>
                            </div>
                            <div class="prose prose-sm max-w-none text-gray-700 dark:text-gray-300">
                                ${formatMessageContent(report.content)}
                            </div>
                        </div>
                    `;
                });
                
                content += '</div></div>';
            }
        });
        
        reportsContent.innerHTML = content;
        
    } catch (error) {
        console.error('Error loading process reports:', error);
        document.getElementById('job-reports-content').innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <p class="text-red-500">Error loading process reports</p>
            </div>
        `;
    }
}

// Agent Statistics Functions
async function loadJobStatistics(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/statistics`);
        const statistics = await response.json();
        
        const statisticsContent = document.getElementById('job-statistics-content');
        
        if (statistics.length === 0) {
            statisticsContent.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-chart-bar text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No agent statistics available yet</p>
                    <p class="text-sm text-gray-400 dark:text-gray-500">Statistics will be collected as agents process your job</p>
                </div>
            `;
            return;
        }
        
        // Calculate totals
        const totalTokens = statistics.reduce((sum, stat) => sum + stat.tokens_consumed, 0);
        const totalCalls = statistics.reduce((sum, stat) => sum + stat.calls_count, 0);
        const totalInputTokens = statistics.reduce((sum, stat) => sum + stat.input_tokens, 0);
        const totalOutputTokens = statistics.reduce((sum, stat) => sum + stat.output_tokens, 0);
        const totalExecutionTime = statistics.reduce((sum, stat) => sum + stat.total_execution_time, 0);
        
        let content = `
            <!-- Summary Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 mb-6">
                <div class="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 p-4 rounded-xl border border-blue-200 dark:border-blue-800">
                    <div class="flex items-center">
                        <i class="fas fa-robot text-2xl text-blue-600 dark:text-blue-400 mr-3"></i>
                        <div>
                            <p class="text-sm text-blue-700 dark:text-blue-300 font-medium">Total Agents</p>
                            <p class="text-2xl font-bold text-blue-900 dark:text-blue-100">${statistics.length}</p>
                        </div>
                    </div>
                </div>
                
                <div class="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/20 p-4 rounded-xl border border-green-200 dark:border-green-800">
                    <div class="flex items-center">
                        <i class="fas fa-phone text-2xl text-green-600 dark:text-green-400 mr-3"></i>
                        <div>
                            <p class="text-sm text-green-700 dark:text-green-300 font-medium">Total Calls</p>
                            <p class="text-2xl font-bold text-green-900 dark:text-green-100">${totalCalls.toLocaleString()}</p>
                        </div>
                    </div>
                </div>
                
                <div class="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/20 p-4 rounded-xl border border-purple-200 dark:border-purple-800">
                    <div class="flex items-center">
                        <i class="fas fa-coins text-2xl text-purple-600 dark:text-purple-400 mr-3"></i>
                        <div>
                            <p class="text-sm text-purple-700 dark:text-purple-300 font-medium">Total Tokens</p>
                            <p class="text-2xl font-bold text-purple-900 dark:text-purple-100">${totalTokens.toLocaleString()}</p>
                        </div>
                    </div>
                </div>
                
                <div class="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/30 dark:to-orange-800/20 p-4 rounded-xl border border-orange-200 dark:border-orange-800">
                    <div class="flex items-center">
                        <i class="fas fa-arrow-down text-2xl text-orange-600 dark:text-orange-400 mr-3"></i>
                        <div>
                            <p class="text-sm text-orange-700 dark:text-orange-300 font-medium">Input Tokens</p>
                            <p class="text-2xl font-bold text-orange-900 dark:text-orange-100">${totalInputTokens.toLocaleString()}</p>
                        </div>
                    </div>
                </div>
                
                <div class="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/30 dark:to-red-800/20 p-4 rounded-xl border border-red-200 dark:border-red-800">
                    <div class="flex items-center">
                        <i class="fas fa-arrow-up text-2xl text-red-600 dark:text-red-400 mr-3"></i>
                        <div>
                            <p class="text-sm text-red-700 dark:text-red-300 font-medium">Output Tokens</p>
                            <p class="text-2xl font-bold text-red-900 dark:text-red-100">${totalOutputTokens.toLocaleString()}</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Detailed Statistics Table -->
            <div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
                <div class="px-6 py-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <i class="fas fa-table mr-2 text-blue-500"></i>Agent Performance Details
                    </h3>
                </div>
                
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-gray-50 dark:bg-gray-800">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Agent Name</th>
                                <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Calls</th>
                                <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Total Tokens</th>
                                <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Input Tokens</th>
                                <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Output Tokens</th>
                                <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Execution Time</th>
                                <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg. Tokens/Call</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
        `;
        
        // Add row for each agent
        statistics.forEach((stat, index) => {
            const avgTokensPerCall = stat.calls_count > 0 ? (stat.tokens_consumed / stat.calls_count).toFixed(1) : '0';
            const executionTimeFormatted = stat.total_execution_time.toFixed(2);
            
            const rowClass = index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800';
            
            content += `
                            <tr class="${rowClass} hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors">
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="flex items-center">
                                        <div class="w-10 h-10 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center text-white font-bold text-sm mr-3">
                                            ${stat.agent_name.charAt(0).toUpperCase()}
                                        </div>
                                        <div>
                                            <div class="text-sm font-medium text-gray-900 dark:text-white">${stat.agent_name}</div>
                                            <div class="text-sm text-gray-500 dark:text-gray-400">Last active: ${new Date(stat.last_updated).toLocaleString()}</div>
                                        </div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-right">
                                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                                        ${stat.calls_count}
                                    </span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-right">
                                    <span class="text-sm font-semibold text-gray-900 dark:text-white">${stat.tokens_consumed.toLocaleString()}</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-right">
                                    <span class="text-sm text-green-600 dark:text-green-400">${stat.input_tokens.toLocaleString()}</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-right">
                                    <span class="text-sm text-red-600 dark:text-red-400">${stat.output_tokens.toLocaleString()}</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-right">
                                    <span class="text-sm text-gray-700 dark:text-gray-300">${executionTimeFormatted}s</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-right">
                                    <span class="text-sm text-purple-600 dark:text-purple-400">${avgTokensPerCall}</span>
                                </td>
                            </tr>
            `;
        });
        
        content += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        statisticsContent.innerHTML = content;
        
    } catch (error) {
        console.error('Error loading agent statistics:', error);
        document.getElementById('job-statistics-content').innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <p class="text-red-500">Error loading agent statistics</p>
                <p class="text-sm text-gray-500 dark:text-gray-400 mt-2">${error.message || 'Unknown error'}</p>
            </div>
        `;
    }
}

// Generated Scripts Functions
async function loadJobScripts(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}/scripts`);
        const scripts = await response.json();
        
        const scriptsContent = document.getElementById('job-scripts-content');
        
        if (scripts.length === 0) {
            scriptsContent.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-code text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No generated scripts available yet</p>
                    <p class="text-sm text-gray-400 dark:text-gray-500">Scripts will appear here as agents generate them</p>
                </div>
            `;
            return;
        }
        
        let content = '<div class="space-y-4">';
        
        scripts.forEach(script => {
            const scriptTypeColors = {
                'training': 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
                'prediction': 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800',
                'visualization': 'bg-pink-50 dark:bg-pink-900/20 border-pink-200 dark:border-pink-800',
                'preprocessing': 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
                'analysis': 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
            };
            
            const scriptTypeIcons = {
                'training': 'fa-brain',
                'prediction': 'fa-chart-line',
                'visualization': 'fa-chart-bar',
                'preprocessing': 'fa-cogs',
                'analysis': 'fa-search'
            };
            
            const typeColor = scriptTypeColors[script.script_type] || 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
            const typeIcon = scriptTypeIcons[script.script_type] || 'fa-file-code';
            
            content += `
                <div class="border border-gray-200 dark:border-gray-700 rounded-lg ${typeColor}" data-script-id="${script.id}">
                    <div class="p-4 border-b border-gray-200 dark:border-gray-700">
                        <div class="flex justify-between items-start">
                            <div>
                                <h4 class="font-semibold text-gray-800 dark:text-white flex items-center">
                                    <i class="fas ${typeIcon} mr-2 text-blue-500"></i>
                                    ${script.script_name}
                                </h4>
                                <div class="flex items-center mt-1 space-x-4 text-sm text-gray-600 dark:text-gray-400">
                                    <span><i class="fas fa-user mr-1"></i>${script.agent_name}</span>
                                    <span><i class="fas fa-tag mr-1"></i>${script.script_type}</span>
                                    <span><i class="fas fa-clock mr-1"></i>${formatDate(script.created_at)}</span>
                                    <span><i class="fas fa-file-alt mr-1"></i>${script.script_content.length.toLocaleString()} chars</span>
                                </div>
                            </div>
                            <div class="flex space-x-2">
                                <button onclick="copyScriptToClipboard('${script.id}')" 
                                        class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm transition-colors">
                                    <i class="fas fa-copy mr-1"></i>Copy
                                </button>
                                <button onclick="downloadScript('${script.id}', '${script.script_name}')" 
                                        class="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded text-sm transition-colors">
                                    <i class="fas fa-download mr-1"></i>Download
                                </button>
                            </div>
                        </div>
                        ${script.execution_result ? `
                        <div class="mt-3 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm">
                            <strong>Execution Result:</strong><br>
                            <span class="font-mono text-xs">${escapeHtml(script.execution_result.substring(0, 200))}${script.execution_result.length > 200 ? '...' : ''}</span>
                        </div>
                        ` : ''}
                    </div>
                    <div class="p-4 bg-gray-900 dark:bg-gray-950 rounded-b-lg">
                        <pre class="text-green-400 text-sm overflow-x-auto"><code class="language-python">${escapeHtml(script.script_content)}</code></pre>
                    </div>
                </div>
            `;
        });
        
        content += '</div>';
        scriptsContent.innerHTML = content;
        
        // Apply syntax highlighting if available
        if (typeof hljs !== 'undefined') {
            setTimeout(() => {
                scriptsContent.querySelectorAll('code').forEach(block => {
                    hljs.highlightElement(block);
                });
            }, 100);
        }
        
    } catch (error) {
        console.error('Error loading generated scripts:', error);
        document.getElementById('job-scripts-content').innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <p class="text-red-500">Error loading generated scripts</p>
            </div>
        `;
    }
}

// Script utility functions
function copyScriptToClipboard(scriptId) {
    // Find the script content
    const scriptElement = document.querySelector(`[data-script-id="${scriptId}"] code`);
    if (scriptElement) {
        navigator.clipboard.writeText(scriptElement.textContent).then(() => {
            showNotification('Script copied to clipboard!', 'success');
        }).catch(() => {
            showNotification('Error copying to clipboard', 'error');
        });
    }
}

function downloadScript(scriptId, scriptName) {
    // Find the script content
    const scriptElement = document.querySelector(`[data-script-id="${scriptId}"] code`);
    if (scriptElement) {
        const blob = new Blob([scriptElement.textContent], { type: 'text/x-python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = scriptName.endsWith('.py') ? scriptName : scriptName + '.py';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showNotification('Script downloaded!', 'success');
    }
}

// ==================== SQL DATASET FUNCTIONS ====================

// Dataset source selection
function selectDatasetSource(source) {
    const fileBtn = document.getElementById('source-file-btn');
    const sqlBtn = document.getElementById('source-sql-btn');
    const fileSection = document.getElementById('file-upload-section');
    const sqlSection = document.getElementById('sql-dataset-section');
    
    if (source === 'file') {
        fileBtn.classList.remove('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        fileBtn.classList.add('bg-purple-500', 'text-white');
        sqlBtn.classList.remove('bg-purple-500', 'text-white');
        sqlBtn.classList.add('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        
        fileSection.classList.remove('hidden');
        sqlSection.classList.add('hidden');
        
        // Reset required attributes
        document.getElementById('dataset-file').required = true;
        document.getElementById('sql-dataset-select').required = false;
    } else {
        sqlBtn.classList.remove('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        sqlBtn.classList.add('bg-purple-500', 'text-white');
        fileBtn.classList.remove('bg-purple-500', 'text-white');
        fileBtn.classList.add('bg-gray-200', 'dark:bg-gray-700', 'text-gray-700', 'dark:text-gray-300');
        
        sqlSection.classList.remove('hidden');
        fileSection.classList.add('hidden');
        
        // Reset required attributes
        document.getElementById('dataset-file').required = false;
        document.getElementById('sql-dataset-select').required = true;
        
        // Load SQL datasets if not already loaded
        loadSqlDatasetOptions();
    }
}

// Load SQL dataset options for job creation
async function loadSqlDatasetOptions() {
    try {
        const response = await fetch('/datasets/sql');
        const datasets = await response.json();
        
        const select = document.getElementById('sql-dataset-select');
        select.innerHTML = '<option value="">Select a SQL dataset</option>';
        
        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = `${dataset.name} (${dataset.row_count} rows, ${dataset.file_size_mb.toFixed(2)} MB)`;
            select.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading SQL datasets:', error);
    }
}

// SQL Dataset management functions
function showCreateDataset() {
    document.getElementById('create-dataset-panel').classList.remove('hidden');
    loadDbConnections();
}

function hideCreateDataset() {
    document.getElementById('create-dataset-panel').classList.add('hidden');
    if (isEditMode) {
        resetEditMode();
    } else {
        clearSqlEditor();
    }
}

function clearSqlEditor() {
    document.getElementById('dataset-name').value = '';
    document.getElementById('dataset-connection').value = '';
    document.getElementById('sql-query').value = '';
    
    // Clear agent fields
    document.getElementById('agent-dataset-connection').value = '';
    document.getElementById('agent-question').value = '';
    document.getElementById('generated-sql-query').value = '';
    
    document.getElementById('query-results').innerHTML = `
        <div class="text-center py-12 text-gray-500 dark:text-gray-400">
            <i class="fas fa-play-circle text-4xl mb-4"></i>
            <p>Click "Preview" or "Generate with AI" to see results</p>
        </div>
    `;
    document.getElementById('query-status').innerHTML = `
        <span class="text-gray-500 dark:text-gray-400">Ready to execute query</span>
    `;
    
    // Reset to manual mode
    selectSQLMode('manual');
}

async function executePreviewQuery() {
    const connectionId = document.getElementById('dataset-connection').value;
    const sqlQuery = document.getElementById('sql-query').value;
    
    if (!connectionId) {
        showNotification('Please select a database connection', 'warning');
        return;
    }
    
    if (!sqlQuery.trim()) {
        showNotification('Please enter a SQL query', 'warning');
        return;
    }
    
    const statusElement = document.getElementById('query-status');
    const resultsElement = document.getElementById('query-results');
    
    try {
        statusElement.innerHTML = '<span class="text-yellow-600"><i class="fas fa-spinner animate-spin mr-2"></i>Executing query...</span>';
        
        const response = await fetch('/sql/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                connection_id: connectionId,
                sql_query: sqlQuery,
                limit: 50
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query execution failed');
        }
        
        const result = await response.json();
        
        statusElement.innerHTML = `<span class="text-green-600"><i class="fas fa-check mr-2"></i>${result.row_count} rows, ${result.column_count} columns</span>`;
        
        // Display results in a table
        let tableHtml = `
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead class="bg-gray-50 dark:bg-gray-800">
                        <tr>
        `;
        
        result.columns.forEach(col => {
            tableHtml += `<th class="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">${col}</th>`;
        });
        
        tableHtml += '</tr></thead><tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">';
        
        result.data.slice(0, 10).forEach((row, index) => {
            tableHtml += `<tr class="${index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}">`;
            result.columns.forEach(col => {
                const value = row[col];
                tableHtml += `<td class="px-4 py-2 text-sm text-gray-900 dark:text-white">${value !== null ? value : '<span class="text-gray-400">null</span>'}</td>`;
            });
            tableHtml += '</tr>';
        });
        
        tableHtml += '</tbody></table>';
        
        if (result.row_count > 10) {
            tableHtml += `<div class="text-center py-2 text-sm text-gray-500">Showing first 10 of ${result.row_count} rows</div>`;
        }
        
        tableHtml += '</div>';
        
        resultsElement.innerHTML = tableHtml;
        
    } catch (error) {
        console.error('Query execution error:', error);
        statusElement.innerHTML = `<span class="text-red-600"><i class="fas fa-exclamation-triangle mr-2"></i>Error</span>`;
        resultsElement.innerHTML = `
            <div class="text-center py-12 text-red-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p class="font-semibold">Query execution failed</p>
                <p class="text-sm mt-2">${error.message}</p>
        `;
    }
}

async function loadSqlDatasets() {
    try {
        const response = await fetch('/datasets/sql');
        const datasets = await response.json();
        
        const container = document.getElementById('sql-datasets-list');
        const countElement = document.getElementById('datasets-count');
        
        countElement.textContent = `${datasets.length} dataset${datasets.length !== 1 ? 's' : ''}`;
        
        if (datasets.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12">
                    <i class="fas fa-database text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                    <p class="text-gray-500 dark:text-gray-400">No SQL datasets created yet</p>
                    <p class="text-sm text-gray-400 dark:text-gray-500">Create your first dataset using SQL queries</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        datasets.forEach(dataset => {
            const createdAt = new Date(dataset.created_at).toLocaleString();
            
            html += `
                <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all duration-300">
                    <div class="flex justify-between items-start mb-4">
                        <div>
                            <h4 class="text-xl font-bold text-gray-900 dark:text-white mb-2">${dataset.name}</h4>
                            <div class="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-300">
                                <span><i class="fas fa-table mr-1"></i>${dataset.row_count.toLocaleString()} rows</span>
                                <span><i class="fas fa-columns mr-1"></i>${dataset.column_count} columns</span>
                                <span><i class="fas fa-weight mr-1"></i>${dataset.file_size_mb.toFixed(2)} MB</span>
                            </div>
                        </div>
                        <div class="flex space-x-2">
                            <button onclick="viewSqlDataset('${dataset.id}')" class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm transition-all">
                                <i class="fas fa-eye mr-1"></i>View
                            </button>
                            <button onclick="editDataset('${dataset.id}', '${dataset.name}')" class="bg-purple-500 hover:bg-purple-600 text-white px-3 py-1 rounded text-sm transition-all">
                                <i class="fas fa-edit mr-1"></i>Edit
                            </button>
                            <button onclick="downloadSqlDataset('${dataset.id}', 'csv')" class="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded text-sm transition-all">
                                <i class="fas fa-download mr-1"></i>CSV
                            </button>
                            <button onclick="downloadSqlDataset('${dataset.id}', 'excel')" class="bg-orange-500 hover:bg-orange-600 text-white px-3 py-1 rounded text-sm transition-all">
                                <i class="fas fa-file-excel mr-1"></i>Excel
                            </button>
                            <button onclick="deleteSqlDataset('${dataset.id}', '${dataset.name}')" class="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-sm transition-all">
                                <i class="fas fa-trash mr-1"></i>Delete
                            </button>
                        </div>
                    </div>
                    
                    <div class="bg-white dark:bg-gray-900 rounded p-3 mb-3">
                        <div class="text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">SQL Query:</div>
                        <code class="text-xs text-gray-600 dark:text-gray-300 font-mono break-all">${dataset.sql_query}</code>
                    </div>
                    
                    <div class="text-xs text-gray-500 dark:text-gray-400">
                        Created: ${createdAt}
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading SQL datasets:', error);
        document.getElementById('sql-datasets-list').innerHTML = `
            <div class="text-center py-12 text-red-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p>Error loading datasets</p>
            </div>
        `;
    }
}

async function viewSqlDataset(datasetId) {
    try {
        const response = await fetch(`/datasets/sql/${datasetId}?limit=100`);
        const result = await response.json();
        
        // Create modal to show data
        showDatasetModal(result);
        
    } catch (error) {
        console.error('Error loading dataset data:', error);
        showNotification('Error loading dataset data', 'error');
    }
}

function showDatasetModal(result) {
    // Create modal HTML
    let modalHtml = `
        <div class="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4">
            <div class="bg-white dark:bg-gray-900 rounded-xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
                <div class="p-6 border-b border-gray-200 dark:border-gray-700">
                    <div class="flex justify-between items-center">
                        <h3 class="text-2xl font-bold text-gray-900 dark:text-white">Dataset Preview</h3>
                        <button onclick="closeDatasetModal()" class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                            <i class="fas fa-times text-2xl"></i>
                        </button>
                    </div>
                    <p class="text-gray-600 dark:text-gray-300 mt-1">
                        Showing ${result.returned_rows} of ${result.total_rows} rows, ${result.column_count} columns
                    </p>
                </div>
                
                <div class="p-6 overflow-auto max-h-[70vh]">
                    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead class="bg-gray-50 dark:bg-gray-800 sticky top-0">
                            <tr>
    `;
    
    result.columns.forEach(col => {
        modalHtml += `<th class="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">${col}</th>`;
    });
    
    modalHtml += '</tr></thead><tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">';
    
    result.data.forEach((row, index) => {
        modalHtml += `<tr class="${index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}">`;
        result.columns.forEach(col => {
            const value = row[col];
            modalHtml += `<td class="px-4 py-2 text-sm text-gray-900 dark:text-white">${value !== null ? value : '<span class="text-gray-400">null</span>'}</td>`;
        });
        modalHtml += '</tr>';
    });
    
    modalHtml += '</tbody></table></div></div></div>';
    
    // Add to DOM
    const modalDiv = document.createElement('div');
    modalDiv.id = 'dataset-modal';
    modalDiv.innerHTML = modalHtml;
    document.body.appendChild(modalDiv);
}

function closeDatasetModal() {
    const modal = document.getElementById('dataset-modal');
    if (modal) {
        modal.remove();
    }
}

async function downloadSqlDataset(datasetId, format) {
    try {
        window.open(`/datasets/sql/${datasetId}/download?format=${format}`, '_blank');
        showNotification(`Dataset download started (${format.toUpperCase()})`, 'success');
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Download failed', 'error');
    }
}

async function deleteSqlDataset(datasetId, datasetName) {
    if (!confirm(`Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/datasets/sql/${datasetId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Delete failed');
        }
        
        showNotification(`Dataset "${datasetName}" deleted successfully`, 'success');
        loadSqlDatasets();
        
    } catch (error) {
        console.error('Delete error:', error);
        showNotification('Error deleting dataset', 'error');
    }
}

// ==================== DATABASE CONFIGURATION FUNCTIONS ====================

async function loadConfiguration() {
    loadDbConnections();
}

async function loadDbConnections() {
    try {
        const response = await fetch('/connections');
        const connections = await response.json();
        
        // Update dataset creation form (manual mode)
        const select = document.getElementById('dataset-connection');
        if (select) {
            select.innerHTML = '<option value="">Select a database connection</option>';
            connections.forEach(conn => {
                const option = document.createElement('option');
                option.value = conn.id;
                option.textContent = `${conn.name} (${conn.db_type})`;
                select.appendChild(option);
            });
        }
        
        // Update dataset creation form (agent mode)
        const agentSelect = document.getElementById('agent-dataset-connection');
        if (agentSelect) {
            agentSelect.innerHTML = '<option value="">Select a database connection</option>';
            connections.forEach(conn => {
                const option = document.createElement('option');
                option.value = conn.id;
                option.textContent = `${conn.name} (${conn.db_type})`;
                agentSelect.appendChild(option);
            });
        }
        
        // Update connections list in configuration
        const container = document.getElementById('connections-list');
        if (container) {
            if (connections.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-12">
                        <i class="fas fa-database text-4xl text-gray-400 dark:text-gray-600 mb-4"></i>
                        <p class="text-gray-500 dark:text-gray-400">No database connections configured</p>
                        <p class="text-sm text-gray-400 dark:text-gray-500">Add your first database connection to start creating SQL datasets</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            connections.forEach(conn => {
                const createdAt = new Date(conn.created_at).toLocaleString();
                
                html += `
                    <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                        <div class="flex justify-between items-start">
                            <div>
                                <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2 flex items-center">
                                    <i class="fas fa-database mr-2 text-blue-500"></i>${conn.name}
                                </h4>
                                <div class="space-y-1 text-sm text-gray-600 dark:text-gray-300">
                                    <div><strong>Type:</strong> ${conn.db_type.charAt(0).toUpperCase() + conn.db_type.slice(1)}</div>
                                    <div><strong>Host:</strong> ${conn.host}:${conn.port}</div>
                                    <div><strong>Database:</strong> ${conn.database_name}</div>
                                    <div><strong>Username:</strong> ${conn.username}</div>
                                    <div><strong>Created:</strong> ${createdAt}</div>
                                </div>
                            </div>
                            <div class="flex space-x-2">
                                <button onclick="editDbConnection('${conn.id}')" class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm transition-all">
                                    <i class="fas fa-edit mr-1"></i>Edit
                                </button>
                                <button onclick="testDbConnection('${conn.id}')" class="bg-yellow-500 hover:bg-yellow-600 text-white px-3 py-1 rounded text-sm transition-all">
                                    <i class="fas fa-plug mr-1"></i>Test
                                </button>
                                <button onclick="deleteDbConnection('${conn.id}', '${conn.name}')" class="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-sm transition-all">
                                    <i class="fas fa-trash mr-1"></i>Delete
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
    } catch (error) {
        console.error('Error loading database connections:', error);
    }
}

function showCreateConnection() {
    // Reset to create mode
    isEditingConnection = false;
    currentEditingConnectionId = null;
    
    // Update UI for create mode
    document.getElementById('connection-form-icon').className = 'fas fa-plus-circle mr-2 text-green-500';
    document.getElementById('connection-form-text').textContent = 'Add New Database Connection';
    document.getElementById('save-connection-icon').className = 'fas fa-save mr-1';
    document.getElementById('save-connection-text').textContent = 'Save Connection';
    
    document.getElementById('create-connection-form').classList.remove('hidden');
}

function hideCreateConnection() {
    document.getElementById('create-connection-form').classList.add('hidden');
    clearConnectionForm();
    
    // Reset editing state
    isEditingConnection = false;
    currentEditingConnectionId = null;
}

function clearConnectionForm() {
    document.getElementById('connection-name').value = '';
    document.getElementById('connection-type').value = 'postgresql';
    document.getElementById('connection-host').value = '';
    document.getElementById('connection-port').value = '';
    document.getElementById('connection-database').value = '';
    document.getElementById('connection-username').value = '';
    document.getElementById('connection-password').value = '';
    document.getElementById('connection-password').placeholder = 'password';
}

async function editDbConnection(connectionId) {
    try {
        // Fetch connection data
        const response = await fetch(`/connections/${connectionId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch connection data');
        }
        
        const connection = await response.json();
        
        // Set editing mode
        isEditingConnection = true;
        currentEditingConnectionId = connectionId;
        
        // Update UI for edit mode
        document.getElementById('connection-form-icon').className = 'fas fa-edit mr-2 text-blue-500';
        document.getElementById('connection-form-text').textContent = 'Edit Database Connection';
        document.getElementById('save-connection-icon').className = 'fas fa-save mr-1';
        document.getElementById('save-connection-text').textContent = 'Update Connection';
        
        // Fill form with existing data
        document.getElementById('connection-name').value = connection.name;
        document.getElementById('connection-type').value = connection.db_type;
        document.getElementById('connection-host').value = connection.host;
        document.getElementById('connection-port').value = connection.port;
        document.getElementById('connection-database').value = connection.database_name;
        document.getElementById('connection-username').value = connection.username;
        document.getElementById('connection-password').value = ''; // Don't show password for security
        document.getElementById('connection-password').placeholder = 'Leave empty to keep current password';
        
        // Show form
        document.getElementById('create-connection-form').classList.remove('hidden');
        
    } catch (error) {
        console.error('Error loading connection for editing:', error);
        showNotification('Failed to load connection data', 'error');
    }
}

async function testConnection() {
    const connectionData = {
        name: document.getElementById('connection-name').value,
        db_type: document.getElementById('connection-type').value,
        host: document.getElementById('connection-host').value,
        port: parseInt(document.getElementById('connection-port').value),
        database_name: document.getElementById('connection-database').value,
        username: document.getElementById('connection-username').value,
        password: document.getElementById('connection-password').value
    };
    
    // Validation
    if (!connectionData.name || !connectionData.host || !connectionData.port || !connectionData.database_name || !connectionData.username || !connectionData.password) {
        showNotification('Please fill in all connection details', 'warning');
        return;
    }
    
    // Show loading state
    showNotification('Testing connection...', 'info');
    
    try {
        const response = await fetch('/connections/test-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(connectionData)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showNotification(result.message, 'success');
        } else {
            showNotification(result.message, 'error');
        }
        
    } catch (error) {
        console.error('Connection test error:', error);
        showNotification('Connection test failed: Network error', 'error');
    }
}

async function testDbConnection(connectionId) {
    try {
        const response = await fetch(`/connections/${connectionId}/test`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showNotification(result.message, 'success');
        } else {
            showNotification(result.message, 'error');
        }
        
    } catch (error) {
        console.error('Connection test error:', error);
        showNotification('Connection test failed', 'error');
    }
}

async function saveConnection() {
    const connectionData = {
        name: document.getElementById('connection-name').value,
        db_type: document.getElementById('connection-type').value,
        host: document.getElementById('connection-host').value,
        port: parseInt(document.getElementById('connection-port').value),
        database_name: document.getElementById('connection-database').value,
        username: document.getElementById('connection-username').value,
        password: document.getElementById('connection-password').value
    };
    
    // Validation (password is optional when editing)
    const passwordRequired = !isEditingConnection;
    if (!connectionData.name || !connectionData.host || !connectionData.port || !connectionData.database_name || !connectionData.username || (passwordRequired && !connectionData.password)) {
        const missingField = passwordRequired ? 'Please fill in all connection details' : 'Please fill in all connection details (password is optional when editing)';
        showNotification(missingField, 'warning');
        return;
    }
    
    try {
        let response;
        let successMessage;
        
        if (isEditingConnection) {
            // Update existing connection
            response = await fetch(`/connections/${currentEditingConnectionId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(connectionData)
            });
            successMessage = `Connection "${connectionData.name}" updated successfully`;
        } else {
            // Create new connection
            response = await fetch('/connections', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(connectionData)
            });
            successMessage = `Connection "${connectionData.name}" created successfully`;
        }
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Connection operation failed');
        }
        
        const result = await response.json();
        showNotification(successMessage, 'success');
        
        hideCreateConnection();
        loadDbConnections();
        
    } catch (error) {
        console.error('Connection save error:', error);
        showNotification(`Error saving connection: ${error.message}`, 'error');
    }
}

async function deleteDbConnection(connectionId, connectionName) {
    if (!confirm(`Are you sure you want to delete the connection "${connectionName}"? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/connections/${connectionId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Delete failed');
        }
        
        showNotification(`Connection "${connectionName}" deleted successfully`, 'success');
        loadDbConnections();
        
    } catch (error) {
        console.error('Delete error:', error);
        showNotification('Error deleting connection', 'error');
    }
}

// Dataset Editing Functions
let currentEditingDataset = null;
let editingData = [];
let isEditMode = false;

async function editDataset(datasetId, datasetName) {
    try {
        // Get dataset metadata first
        const metadataResponse = await fetch('/datasets/sql');
        const datasets = await metadataResponse.json();
        const datasetInfo = datasets.find(ds => ds.id === datasetId);
        
        if (!datasetInfo) {
            throw new Error('Dataset not found');
        }
        
        // Load dataset data
        const response = await fetch(`/datasets/sql/${datasetId}`);
        if (!response.ok) {
            throw new Error('Failed to load dataset');
        }
        
        const dataset = await response.json();
        currentEditingDataset = datasetId;
        window.currentDatasetId = datasetId; // Sync both variables
        editingData = [...dataset.data]; // Create copy
        isEditMode = true;
        
        // Switch to edit mode UI
        setupEditMode(datasetName, datasetInfo, dataset);
        
        // Show the create dataset panel (now in edit mode)
        document.getElementById('create-dataset-panel').classList.remove('hidden');
        
        // Update status
        updateQueryStatus(`Loaded ${dataset.total_rows} rows for editing`);
        
    } catch (error) {
        console.error('Error loading dataset for editing:', error);
        showNotification('Failed to load dataset for editing', 'error');
    }
}

function setupEditMode(datasetName, datasetInfo, dataset) {
    // Update panel title and appearance
    document.getElementById('dataset-panel-text').textContent = `Edit Dataset: ${datasetName}`;
    document.getElementById('dataset-panel-icon').className = 'fas fa-edit mr-2 text-purple-500';
    
    // Update results panel
    document.getElementById('results-panel-text').textContent = 'Dataset Editor';
    document.getElementById('results-panel-icon').className = 'fas fa-edit mr-2 text-purple-500';
    
    // Update save button
    const manualSaveBtn = document.getElementById('manual-save-btn');
    if (manualSaveBtn) {
        manualSaveBtn.innerHTML = '<i class="fas fa-save mr-2"></i>Save Changes';
    }
    
    // Show edit actions
    document.getElementById('edit-actions').classList.remove('hidden');
    
    // Check if this was generated with agent
    if (datasetInfo.generation_type === 'agent' && datasetInfo.agent_prompt) {
        // Configure for agent mode editing
        selectSQLMode('agent');
        
        // Set the current dataset ID for regeneration
        window.currentDatasetId = datasetInfo.id;
        
        // Fill agent form with existing data
        document.getElementById('dataset-name').value = datasetName;
        document.getElementById('agent-dataset-connection').value = datasetInfo.connection_id;
        document.getElementById('agent-question').value = datasetInfo.agent_prompt;
        document.getElementById('generated-sql-query').value = datasetInfo.sql_query;
        
        // Update generate button to show it's in edit mode
        const generateBtn = document.getElementById('agent-generate-btn');
        generateBtn.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Regenerate with AI';
        
        // Load connections for agent mode
        loadDbConnections().then(() => {
            document.getElementById('agent-dataset-connection').value = datasetInfo.connection_id;
        });
    } else {
        // Force manual mode for manual datasets
        selectSQLMode('manual');
        
        // Fill manual form with existing data
        document.getElementById('dataset-name').value = datasetName;
        document.getElementById('dataset-connection').value = datasetInfo.connection_id;
        document.getElementById('sql-query').value = datasetInfo.sql_query;
        
        // Load connections for manual mode
        loadDbConnections().then(() => {
            document.getElementById('dataset-connection').value = datasetInfo.connection_id;
        });
    }
    
    // Generate editable table
    renderEditableTableInResults(dataset.columns, editingData);
}

function renderEditableTableInResults(columns, data) {
    if (!columns || columns.length === 0) {
        document.getElementById('query-results').innerHTML = '<div class="text-center py-8 text-gray-500">No data available</div>';
        return;
    }
    
    let tableHtml = `
        <table class="min-w-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
            <thead class="bg-gray-50 dark:bg-gray-800">
                <tr>
                    <th class="px-2 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider border-r border-gray-200 dark:border-gray-700">#</th>
    `;
    
    // Add column headers
    columns.forEach(column => {
        tableHtml += `<th class="px-2 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider border-r border-gray-200 dark:border-gray-700">${column}</th>`;
    });
    
    tableHtml += `
                    <th class="px-2 py-2 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Actions</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
    `;
    
    // Add data rows
    data.forEach((row, rowIndex) => {
        tableHtml += `<tr class="hover:bg-gray-50 dark:hover:bg-gray-800">`;
        
        // Row number (clickable for delete)
        tableHtml += `
            <td class="px-2 py-2 text-sm text-gray-500 dark:text-gray-300 border-r border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-red-50 dark:hover:bg-red-900/20" 
                onclick="deleteEditRow(${rowIndex})" title="Click to delete row">
                ${rowIndex + 1}
            </td>
        `;
        
        // Data cells (editable)
        columns.forEach(column => {
            const cellValue = row[column] || '';
            tableHtml += `
                <td class="px-2 py-2 text-sm text-gray-900 dark:text-gray-100 border-r border-gray-200 dark:border-gray-700">
                    <div class="editable-cell cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-900/20 p-1 rounded" 
                         data-row="${rowIndex}" 
                         data-column="${column}" 
                         ondblclick="editCell(this)">${cellValue}</div>
                </td>
            `;
        });
        
        // Actions
        tableHtml += `
            <td class="px-2 py-2 text-center">
                <button onclick="deleteEditRow(${rowIndex})" class="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 text-xs" title="Delete row">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        `;
        
        tableHtml += '</tr>';
    });
    
    tableHtml += '</tbody></table>';
    
    document.getElementById('query-results').innerHTML = tableHtml;
}

function editCell(cellElement) {
    const currentValue = cellElement.textContent;
    const rowIndex = parseInt(cellElement.dataset.row);
    const column = cellElement.dataset.column;
    
    // Create input element
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentValue;
    input.className = 'w-full px-1 py-0 text-sm bg-white dark:bg-gray-700 border border-blue-500 rounded';
    
    // Save on blur or Enter
    function saveCell() {
        const newValue = input.value;
        cellElement.textContent = newValue;
        
        // Update data in memory
        editingData[rowIndex][column] = newValue;
        
        updateQueryStatus(`Updated cell at row ${rowIndex + 1}, column ${column}`);
    }
    
    input.addEventListener('blur', saveCell);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            saveCell();
        } else if (e.key === 'Escape') {
            cellElement.textContent = currentValue; // Restore original value
        }
    });
    
    // Replace cell content with input
    cellElement.innerHTML = '';
    cellElement.appendChild(input);
    input.focus();
    input.select();
}

function deleteEditRow(rowIndex) {
    if (!confirm(`Are you sure you want to delete row ${rowIndex + 1}?`)) {
        return;
    }
    
    // Remove from data
    editingData.splice(rowIndex, 1);
    
    // Re-render table
    if (editingData.length > 0) {
        const columns = Object.keys(editingData[0]);
        renderEditableTableInResults(columns, editingData);
    } else {
        document.getElementById('query-results').innerHTML = '<div class="text-center py-8 text-gray-500">No data remaining</div>';
    }
    
    updateQueryStatus(`Deleted row ${rowIndex + 1}. ${editingData.length} rows remaining.`);
}

function addNewRowToEdit() {
    if (editingData.length === 0) {
        showNotification('Cannot add row to empty dataset', 'warning');
        return;
    }
    
    // Create new row with empty values
    const columns = Object.keys(editingData[0]);
    const newRow = {};
    columns.forEach(col => newRow[col] = '');
    
    editingData.push(newRow);
    
    // Re-render table
    renderEditableTableInResults(columns, editingData);
    
    updateQueryStatus(`Added new row. Total: ${editingData.length} rows.`);
}

// Update the existing executeAndSaveDataset function to handle both create and edit modes
async function executeAndSaveDataset() {
    if (isEditMode) {
        // Save existing dataset changes
        await saveDatasetChanges();
    } else {
        // Create new dataset (existing functionality)
        await createNewDataset();
    }
}

async function saveDatasetChanges() {
    if (!currentEditingDataset || editingData.length === 0) {
        showNotification('No changes to save', 'warning');
        return;
    }
    
    try {
        updateQueryStatus('Saving changes...');
        
        const response = await fetch(`/datasets/sql/${currentEditingDataset}/edit`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                data: editingData
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Save failed');
        }
        
        const result = await response.json();
        
        showNotification('Dataset updated successfully!', 'success');
        updateQueryStatus(`Saved ${result.row_count} rows, ${result.column_count} columns`);
        
        // Refresh dataset list and exit edit mode
        loadSqlDatasets();
        resetEditMode();
        
    } catch (error) {
        console.error('Error saving dataset:', error);
        showNotification(`Error saving dataset: ${error.message}`, 'error');
        updateQueryStatus('Save failed');
    }
}

async function createNewDataset() {
    // This is the existing dataset creation logic
    const datasetName = document.getElementById('dataset-name').value;
    const connectionId = document.getElementById('dataset-connection').value;
    const sqlQuery = document.getElementById('sql-query').value;
    
    // Validation
    if (!datasetName || !connectionId || !sqlQuery) {
        showNotification('Please fill in all dataset details', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/datasets/sql', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: datasetName,
                connection_id: connectionId,
                sql_query: sqlQuery
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Dataset creation failed');
        }
        
        const result = await response.json();
        showNotification(`Dataset "${datasetName}" created successfully`, 'success');
        
        clearSqlEditor();
        loadSqlDatasets();
        
    } catch (error) {
        console.error('Dataset creation error:', error);
        showNotification(`Error creating dataset: ${error.message}`, 'error');
    }
}

function resetEditMode() {
    isEditMode = false;
    currentEditingDataset = null;
    window.currentDatasetId = null; // Also clear this
    editingData = [];
    
    // Clear current dataset ID for agent regeneration
    window.currentDatasetId = null;
    
    // Reset UI to create mode
    document.getElementById('dataset-panel-text').textContent = 'Create Dataset with SQL';
    document.getElementById('dataset-panel-icon').className = 'fas fa-code mr-2 text-purple-500';
    document.getElementById('results-panel-text').textContent = 'Query Results Preview';
    document.getElementById('results-panel-icon').className = 'fas fa-table mr-2 text-blue-500';
    
    // Reset manual save button
    const manualSaveBtn = document.getElementById('manual-save-btn');
    if (manualSaveBtn) {
        manualSaveBtn.innerHTML = '<i class="fas fa-save mr-2"></i>Save Dataset';
    }
    
    // Reset agent generate button
    const agentGenerateBtn = document.getElementById('agent-generate-btn');
    if (agentGenerateBtn) {
        agentGenerateBtn.innerHTML = '<i class="fas fa-magic mr-2"></i>Generate with AI';
    }
    
    // Remove any preview buttons
    const previewBtn = document.getElementById('preview-dataset-btn');
    if (previewBtn) {
        previewBtn.remove();
    }
    
    document.getElementById('edit-actions').classList.add('hidden');
    
    // Clear form
    clearSqlEditor();
}

function updateQueryStatus(message) {
    const statusElement = document.getElementById('query-status');
    if (statusElement) {
        statusElement.innerHTML = `<span class="text-gray-500 dark:text-gray-400">${message}</span>`;
    }
}

// ==================== JOB VERSIONING FUNCTIONS ====================
let currentVersionParentId = null;

async function showVersionsModal(jobId) {
    currentVersionParentId = jobId;
    
    // Show modal
    document.getElementById('versions-modal').classList.remove('hidden');
    
    // Load versions
    await loadJobVersions(jobId);
    
    // Set up form handler
    document.getElementById('create-version-form').onsubmit = async (e) => {
        e.preventDefault();
        await createJobVersion();
    };
}

function closeVersionsModal() {
    document.getElementById('versions-modal').classList.add('hidden');
    currentVersionParentId = null;
    
    // Clear form
    document.getElementById('version-name').value = '';
    document.getElementById('version-prompt').value = '';
    document.getElementById('version-comparison').classList.add('hidden');
}

async function loadJobVersions(parentJobId) {
    try {
        const response = await fetch(`/jobs/${parentJobId}/versions`);
        const versions = await response.json();
        
        displayVersionsList(versions);
        
    } catch (error) {
        console.error('Error loading job versions:', error);
        showNotification('Error loading job versions', 'error');
    }
}

function displayVersionsList(versions) {
    const versionsList = document.getElementById('versions-list');
    
    if (versions.length === 0) {
        versionsList.innerHTML = `
            <div class="text-center py-8 text-gray-500 dark:text-gray-400">
                <i class="fas fa-code-branch text-4xl mb-4"></i>
                <p>No versions created yet</p>
            </div>
        `;
        return;
    }
    
    const versionsHTML = versions.map(version => {
        const statusColors = {
            'completed': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
            'failed': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
            'processing': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
            'created': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
        };
        
        const statusIcons = {
            'completed': 'fa-check-circle',
            'failed': 'fa-times-circle',
            'processing': 'fa-spinner fa-spin',
            'created': 'fa-clock'
        };
        
        return `
            <div class="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center">
                        <span class="font-medium text-gray-900 dark:text-white mr-2">
                            ${version.name}
                        </span>
                        <span class="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 px-2 py-1 rounded text-xs">
                            v${version.version_number}
                        </span>
                        ${version.is_parent ? '<span class="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded text-xs ml-1">Original</span>' : ''}
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${statusColors[version.status] || 'bg-gray-100 text-gray-800'}">
                            <i class="fas ${statusIcons[version.status] || 'fa-question'} mr-1"></i>
                            ${version.status}
                        </span>
                    </div>
                </div>
                
                <div class="text-sm text-gray-600 dark:text-gray-300 mb-3">
                    <div class="truncate">
                        <strong>Prompt:</strong> ${version.prompt.length > 100 ? version.prompt.substring(0, 100) + '...' : version.prompt}
                    </div>
                </div>
                
                <div class="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-3">
                    <span><i class="fas fa-calendar mr-1"></i>${new Date(version.created_at).toLocaleString()}</span>
                </div>
                
                <div class="flex space-x-2">
                    ${!version.is_parent && version.status !== 'processing' ? `
                        <button onclick="runJobVersion('${version.id}')" 
                                class="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded text-xs transition-all">
                            <i class="fas fa-play mr-1"></i>Run
                        </button>
                    ` : ''}
                    
                    <button onclick="openJobDetails('${version.id}')" 
                            class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-xs transition-all">
                        <i class="fas fa-eye mr-1"></i>View
                    </button>
                    
                    ${!version.is_parent ? `
                        <button onclick="deleteJobVersion('${version.id}')" 
                                class="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-xs transition-all">
                            <i class="fas fa-trash mr-1"></i>Delete
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
    
    versionsList.innerHTML = versionsHTML;
    
    // Add compare button if there are multiple versions
    if (versions.length > 1) {
        const compareButton = `
            <div class="mt-4 text-center">
                <button onclick="compareVersions('${currentVersionParentId}')" 
                        class="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white px-6 py-2 rounded-lg transition-all duration-300">
                    <i class="fas fa-balance-scale mr-2"></i>Compare All Versions
                </button>
            </div>
        `;
        versionsList.innerHTML += compareButton;
    }
}

async function createJobVersion() {
    const name = document.getElementById('version-name').value;
    const prompt = document.getElementById('version-prompt').value;
    
    if (!name.trim() || !prompt.trim()) {
        showNotification('Please fill in all fields', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/jobs/${currentVersionParentId}/versions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                parent_job_id: currentVersionParentId,
                name: name,
                prompt: prompt
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification(`Version ${result.version_number} created successfully!`, 'success');
            
            // Clear form
            document.getElementById('version-name').value = '';
            document.getElementById('version-prompt').value = '';
            
            // Reload versions list
            await loadJobVersions(currentVersionParentId);
            
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create version');
        }
        
    } catch (error) {
        console.error('Error creating version:', error);
        showNotification(`Error creating version: ${error.message}`, 'error');
    }
}

async function runJobVersion(versionJobId) {
    if (!confirm('Are you sure you want to run this version? This will start the machine learning process.')) {
        return;
    }
    
    try {
        const response = await fetch(`/jobs/versions/${versionJobId}/run`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification(result.message, 'success');
            
            // Close modal and refresh jobs list
            closeVersionsModal();
            if (typeof loadJobs === 'function') {
                loadJobs();
            }
            
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to run version');
        }
        
    } catch (error) {
        console.error('Error running version:', error);
        showNotification(`Error running version: ${error.message}`, 'error');
    }
}

async function deleteJobVersion(versionJobId) {
    if (!confirm('Are you sure you want to delete this version? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`/jobs/versions/${versionJobId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('Version deleted successfully', 'success');
            
            // Reload versions list
            await loadJobVersions(currentVersionParentId);
            
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete version');
        }
        
    } catch (error) {
        console.error('Error deleting version:', error);
        showNotification(`Error deleting version: ${error.message}`, 'error');
    }
}

async function compareVersions(parentJobId) {
    try {
        const response = await fetch(`/jobs/${parentJobId}/versions/comparison`);
        const comparison = await response.json();
        
        displayVersionComparison(comparison);
        
    } catch (error) {
        console.error('Error comparing versions:', error);
        showNotification('Error loading version comparison', 'error');
    }
}

function displayVersionComparison(comparison) {
    const comparisonSection = document.getElementById('version-comparison');
    const comparisonContent = document.getElementById('comparison-content');
    
    const versionsHTML = comparison.versions.map(version => `
        <div class="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 mb-4">
            <div class="flex items-center justify-between mb-2">
                <h5 class="font-semibold text-gray-900 dark:text-white">
                    ${version.name} <span class="text-sm text-gray-500">v${version.version_number}</span>
                </h5>
                <span class="px-2 py-1 rounded text-xs ${
                    version.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    version.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                    version.status === 'processing' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                    'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                }">
                    ${version.status}
                </span>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                    <strong class="text-gray-700 dark:text-gray-300">Prompt:</strong>
                    <p class="text-gray-600 dark:text-gray-400 mt-1">${version.prompt}</p>
                </div>
                
                <div>
                    <strong class="text-gray-700 dark:text-gray-300">Models:</strong>
                    <p class="text-gray-600 dark:text-gray-400 mt-1">
                        ${version.model_count} model(s) generated
                        ${version.model_names.length ? `<br><span class="text-xs">${version.model_names.join(', ')}</span>` : ''}
                    </p>
                </div>
                
                <div>
                    <strong class="text-gray-700 dark:text-gray-300">Agent Statistics:</strong>
                    <div class="mt-1">
                        ${version.agent_statistics.map(stat => `
                            <div class="text-xs text-gray-600 dark:text-gray-400">
                                ${stat.agent_name}: ${stat.total_calls} calls, ${stat.total_tokens} tokens
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
            
            <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
                Created: ${new Date(version.created_at).toLocaleString()}
            </div>
        </div>
    `).join('');
    
    comparisonContent.innerHTML = `
        <div class="mb-4">
            <h5 class="font-semibold text-gray-900 dark:text-white mb-2">
                Comparing ${comparison.total_versions} versions
            </h5>
        </div>
        ${versionsHTML}
    `;
    
    comparisonSection.classList.remove('hidden');
}

// ==================== NEW VERSION NAVIGATION FUNCTIONS ====================

async function createVersionFromJob(jobId) {
    try {
        // Primero obtenemos la información del trabajo para determinar si es padre o hijo
        const response = await fetch(`/jobs/${jobId}`);
        const job = await response.json();
        
        // If it's a child, we use the parent's ID; if it's a parent, we use its own ID
        const parentJobId = job.parent_job_id || jobId;
        
        // Show modal to create new version
        currentVersionParentId = parentJobId;
        document.getElementById('versions-modal').classList.remove('hidden');
        
        // Load existing versions
        await loadJobVersions(parentJobId);
        
        // Focus the name field to facilitate entry
        setTimeout(() => {
            document.getElementById('version-name').focus();
        }, 100);
        
        // Pre-fill with current job information if different from parent
        if (job.parent_job_id) {
            document.getElementById('version-name').value = `${job.name} - Enhanced`;
            document.getElementById('version-prompt').value = job.prompt;
        }
        
    } catch (error) {
        console.error('Error accessing job for version creation:', error);
        showNotification('Error accessing job information', 'error');
    }
}

async function showChildVersions(parentJobId) {
    try {
        const response = await fetch(`/jobs/${parentJobId}/versions`);
        const versions = await response.json();
        
        // Filtrar solo las versiones hijas (excluir el padre)
        const childVersions = versions.filter(v => !v.is_parent);
        
        if (childVersions.length === 0) {
            showNotification('Este trabajo no tiene versiones hijas aún', 'info');
            return;
        }
        
        // Mostrar modal con las versiones hijas
        showVersionsList(childVersions, `Versiones Hijas (${childVersions.length})`);
        
    } catch (error) {
        console.error('Error loading child versions:', error);
        showNotification('Error loading child versions', 'error');
    }
}

async function showParentVersion(parentJobId) {
    if (!parentJobId) {
        showNotification('No se encontró el trabajo padre', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/jobs/${parentJobId}`);
        const parentJob = await response.json();
        
        // Mostrar modal con información del padre
        showJobInfo(parentJob, 'Trabajo Padre');
        
    } catch (error) {
        console.error('Error loading parent job:', error);
        showNotification('Error loading parent job', 'error');
    }
}

async function showSiblingVersions(parentJobId, currentJobId) {
    if (!parentJobId) {
        showNotification('No se encontró el trabajo padre', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/jobs/${parentJobId}/versions`);
        const versions = await response.json();
        
        // Destacar la versión actual
        versions.forEach(v => {
            v.isCurrent = v.id === currentJobId;
        });
        
        // Mostrar modal con todas las versiones
        showVersionsList(versions, `Todas las Versiones (${versions.length})`);
        
    } catch (error) {
        console.error('Error loading sibling versions:', error);
        showNotification('Error loading versions', 'error');
    }
}

function showVersionsList(versions, title) {
    // Create a custom modal to show the versions list
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4';
    modal.innerHTML = `
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
            <div class="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700">
                <h3 class="text-xl font-bold text-gray-900 dark:text-white">
                    <i class="fas fa-list mr-2 text-blue-500"></i>${title}
                </h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            
            <div class="p-6 overflow-y-auto max-h-[60vh]">
                <div class="space-y-4">
                    ${versions.map(version => {
                        const statusColors = {
                            'completed': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
                            'failed': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
                            'processing': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
                            'created': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                        };
                        
                        return `
                            <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 ${version.isCurrent ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'bg-white dark:bg-gray-900'}">
                                <div class="flex items-center justify-between mb-3">
                                    <div class="flex items-center space-x-2">
                                        <h4 class="font-semibold text-gray-900 dark:text-white">${version.name}</h4>
                                        <span class="px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
                                            v${version.version_number}
                                        </span>
                                        ${version.is_parent ? '<span class="px-2 py-1 rounded text-xs bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200">Original</span>' : ''}
                                        ${version.isCurrent ? '<span class="px-2 py-1 rounded text-xs bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200">Actual</span>' : ''}
                                    </div>
                                    <span class="px-2 py-1 rounded-full text-xs font-medium ${statusColors[version.status] || 'bg-gray-100 text-gray-800'}">
                                        ${version.status}
                                    </span>
                                </div>
                                
                                <p class="text-sm text-gray-600 dark:text-gray-300 mb-3">${version.prompt.substring(0, 120)}${version.prompt.length > 120 ? '...' : ''}</p>
                                
                                <div class="flex items-center justify-between">
                                    <div class="text-xs text-gray-500 dark:text-gray-400">
                                        Creado: ${new Date(version.created_at).toLocaleString()}
                                    </div>
                                    <div class="space-x-2">
                                        <button onclick="openJobDetails('${version.id}'); this.closest('.fixed').remove();" 
                                                class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-xs">
                                            <i class="fas fa-eye mr-1"></i>Ver
                                        </button>
                                        ${!version.is_parent && version.status !== 'processing' ? `
                                            <button onclick="runJobVersion('${version.id}'); this.closest('.fixed').remove();" 
                                                    class="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded text-xs">
                                                <i class="fas fa-play mr-1"></i>Ejecutar
                                            </button>
                                        ` : ''}
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
                
                <div class="mt-6 text-center">
                    <button onclick="createVersionFromJob('${versions[0].parent_job_id || versions[0].id}'); this.closest('.fixed').remove();" 
                            class="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-6 py-2 rounded-lg">
                        <i class="fas fa-plus mr-2"></i>Create New Version
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

function showJobInfo(job, title) {
    // Create a modal to show job information
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4';
    modal.innerHTML = `
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div class="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700">
                <h3 class="text-xl font-bold text-gray-900 dark:text-white">
                    <i class="fas fa-info-circle mr-2 text-blue-500"></i>${title}
                </h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            
            <div class="p-6">
                <div class="space-y-4">
                    <div>
                        <h4 class="font-semibold text-gray-900 dark:text-white mb-2">${job.name}</h4>
                        <div class="flex items-center space-x-2 mb-2">
                            <span class="px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-700">v${job.version_number}</span>
                            <span class="px-2 py-1 rounded text-xs ${job.status === 'completed' ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200' : 'bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200'}">
                                ${job.status}
                            </span>
                        </div>
                    </div>
                    
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Objective:</p>
                        <p class="text-sm text-gray-800 dark:text-gray-200 bg-gray-50 dark:bg-gray-700 p-3 rounded">${job.prompt}</p>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <p class="text-sm font-medium text-gray-600 dark:text-gray-300">Target column:</p>
                            <p class="text-sm text-gray-800 dark:text-gray-200">${job.target_column}</p>
                        </div>
                        <div>
                            <p class="text-sm font-medium text-gray-600 dark:text-gray-300">Progress:</p>
                            <p class="text-sm text-gray-800 dark:text-gray-200">${job.progress}%</p>
                        </div>
                    </div>
                    
                    <div>
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-300">Created:</p>
                        <p class="text-sm text-gray-800 dark:text-gray-200">${new Date(job.created_at).toLocaleString()}</p>
                    </div>
                </div>
                
                <div class="mt-6 flex space-x-3">
                    <button onclick="openJobDetails('${job.id}'); this.closest('.fixed').remove();" 
                            class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
                        <i class="fas fa-eye mr-2"></i>View Details
                    </button>
                    <button onclick="createVersionFromJob('${job.id}'); this.closest('.fixed').remove();" 
                            class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded">
                        <i class="fas fa-plus mr-2"></i>Create Version
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

// SQL Agent Functions
function selectSQLMode(mode) {
    const manualBtn = document.getElementById('mode-manual-btn');
    const agentBtn = document.getElementById('mode-agent-btn');
    const manualFields = document.getElementById('manual-mode-fields');
    const agentFields = document.getElementById('agent-mode-fields');
    const manualSqlSection = document.getElementById('manual-sql-section');
    const agentSqlSection = document.getElementById('agent-sql-section');
    const manualSaveBtn = document.getElementById('manual-save-btn');
    const agentGenerateBtn = document.getElementById('agent-generate-btn');
    const agentSaveEditedBtn = document.getElementById('agent-save-edited-btn');

    if (mode === 'manual') {
        // Update button styles
        manualBtn.className = 'flex-1 px-4 py-3 bg-purple-500 text-white font-medium';
        agentBtn.className = 'flex-1 px-4 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium';
        
        // Show manual fields, hide agent fields
        manualFields.classList.remove('hidden');
        agentFields.classList.add('hidden');
        manualSqlSection.classList.remove('hidden');
        agentSqlSection.classList.add('hidden');
        manualSaveBtn.classList.remove('hidden');
        agentGenerateBtn.classList.add('hidden');
        agentSaveEditedBtn.classList.add('hidden');
    } else if (mode === 'agent') {
        // Update button styles
        manualBtn.className = 'flex-1 px-4 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium';
        agentBtn.className = 'flex-1 px-4 py-3 bg-indigo-500 text-white font-medium';
        
        // Show agent fields, hide manual fields
        manualFields.classList.add('hidden');
        agentFields.classList.remove('hidden');
        manualSqlSection.classList.add('hidden');
        agentSqlSection.classList.remove('hidden');
        manualSaveBtn.classList.add('hidden');
        agentGenerateBtn.classList.remove('hidden');
        agentSaveEditedBtn.classList.remove('hidden');
    }
}

async function generateDatasetWithAgent() {
    const name = document.getElementById('dataset-name').value.trim();
    const question = document.getElementById('agent-question').value.trim();
    const connectionId = document.getElementById('agent-dataset-connection').value.trim();
    
    if (!name) {
        showNotification('Please enter a dataset name', 'warning');
        return;
    }
    
    if (!question) {
        showNotification('Please describe what data you need', 'warning');
        return;
    }
    
    if (!connectionId) {
        showNotification('Please select a database connection', 'warning');
        return;
    }
    
    const generateBtn = document.getElementById('agent-generate-btn');
    const originalText = generateBtn.innerHTML;
    
    try {
        // Update button to show loading state
        generateBtn.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i>Generating...';
        generateBtn.disabled = true;
        
        // Prepare request data (include dataset_id if updating)
        const requestData = {
            name: name,
            question: question,
            connection_id: connectionId
        };
        
        // If we're regenerating an existing dataset, include the dataset_id
        if (window.currentDatasetId) {
            requestData.dataset_id = window.currentDatasetId;
        }
        
        // Call the SQL agent endpoint
        const response = await fetch('/datasets/sql/agent-generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to generate dataset');
        }
        
        // Display the generated SQL in the editable textarea
        document.getElementById('generated-sql-query').value = result.sql_query;
        
        // Store original SQL for reset functionality
        originalGeneratedSQL = result.sql_query;
        
        // Update query results section
        const resultsElement = document.getElementById('query-results');
        resultsElement.innerHTML = `
            <div class="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
                <div class="flex items-start">
                    <div class="flex-shrink-0">
                        <i class="fas fa-check-circle text-green-400 text-xl"></i>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-green-800 dark:text-green-300">Dataset Generated Successfully!</h3>
                        <div class="mt-2 text-sm text-green-700 dark:text-green-400">
                            <p><strong>Rows:</strong> ${result.row_count}</p>
                            <p><strong>Columns:</strong> ${result.column_count}</p>
                            <p><strong>Size:</strong> ${result.file_size_mb.toFixed(2)} MB</p>
                            <p><strong>Columns:</strong> ${result.columns.join(', ')}</p>
                            ${result.is_mock ? '<p class="mt-2 text-yellow-600 dark:text-yellow-400"><i class="fas fa-exclamation-triangle mr-1"></i>Note: This is mock data (database not available)</p>' : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Update status
        document.getElementById('query-status').innerHTML = `
            <span class="text-green-600"><i class="fas fa-check mr-2"></i>Dataset created successfully</span>
        `;
        
        // Show success notification
        showNotification('Dataset generated successfully with AI Agent!', 'success');
        
        // Store the current dataset ID for potential updates
        window.currentDatasetId = result.id;
        
        // Change the generate button to "Regenerate with AI" for iterations
        generateBtn.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Regenerate with AI';
        
        // Add preview button
        const previewBtn = document.createElement('button');
        previewBtn.className = 'ml-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600';
        previewBtn.innerHTML = '<i class="fas fa-eye mr-2"></i>Preview Data';
        previewBtn.onclick = () => previewDataset(result.id);
        
        if (!document.getElementById('preview-dataset-btn')) {
            previewBtn.id = 'preview-dataset-btn';
            generateBtn.parentNode.insertBefore(previewBtn, generateBtn.nextSibling);
        }
        
        // Refresh the datasets list
        await loadSqlDatasets();
        
        // Keep dialog open for iterations - don't close automatically
        
    } catch (error) {
        console.error('Error generating dataset with agent:', error);
        showNotification(error.message || 'Failed to generate dataset with AI Agent', 'error');
        
        // Show error in results
        document.getElementById('query-results').innerHTML = `
            <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <div class="flex items-start">
                    <div class="flex-shrink-0">
                        <i class="fas fa-exclamation-circle text-red-400 text-xl"></i>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800 dark:text-red-300">Generation Failed</h3>
                        <div class="mt-2 text-sm text-red-700 dark:text-red-400">
                            <p>${error.message || 'Unknown error occurred'}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } finally {
        // Restore button state (but keep updated text if successful)
        generateBtn.disabled = false;
        if (!window.currentDatasetId) {
            generateBtn.innerHTML = originalText;
        }
    }
}

async function previewDataset(datasetId) {
    try {
        const response = await fetch(`/datasets/sql/${datasetId}?limit=10`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load dataset');
        }
        
        // Create preview modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4';
        modal.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden">
                <div class="p-6 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <h2 class="text-xl font-semibold">Dataset Preview</h2>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                <div class="p-6 overflow-auto max-h-[70vh]">
                    <div class="mb-4">
                        <p><strong>Total Rows:</strong> ${data.total_rows}</p>
                        <p><strong>Columns:</strong> ${data.column_count}</p>
                        <p class="text-sm text-gray-600">Showing first ${data.returned_rows} rows</p>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
                            <thead>
                                <tr class="bg-gray-50 dark:bg-gray-700">
                                    ${data.columns.map(col => `<th class="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${data.data.map(row => 
                                    `<tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
                                        ${data.columns.map(col => `<td class="border border-gray-300 dark:border-gray-600 px-4 py-2">${row[col] || ''}</td>`).join('')}
                                    </tr>`
                                ).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('Error previewing dataset:', error);
        showNotification('Failed to preview dataset', 'error');
    }
}

// Store original generated SQL for reset functionality
let originalGeneratedSQL = '';

// Function to display query results in the agent dataset preview area
function displayQueryResults(result) {
    const resultsElement = document.getElementById('agent-dataset-preview');
    
    if (!resultsElement) {
        console.error('Agent dataset preview element not found');
        return;
    }
    
    try {
        // Display results in a table
        let tableHtml = `
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead class="bg-gray-50 dark:bg-gray-800">
                        <tr>
        `;
        
        result.columns.forEach(col => {
            tableHtml += `<th class="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">${col}</th>`;
        });
        
        tableHtml += '</tr></thead><tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">';
        
        result.data.slice(0, 10).forEach((row, index) => {
            tableHtml += `<tr class="${index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}">`;
            result.columns.forEach(col => {
                const value = row[col];
                tableHtml += `<td class="px-4 py-2 text-sm text-gray-900 dark:text-white">${value !== null ? value : '<span class="text-gray-400">null</span>'}</td>`;
            });
            tableHtml += '</tr>';
        });
        
        tableHtml += '</tbody></table>';
        
        if (result.row_count > 10) {
            tableHtml += `<div class="text-center py-2 text-sm text-gray-500">Showing first 10 of ${result.row_count} rows</div>`;
        }
        
        tableHtml += '</div>';
        
        resultsElement.innerHTML = tableHtml;
        
    } catch (error) {
        console.error('Error displaying query results:', error);
        resultsElement.innerHTML = `
            <div class="text-center py-12 text-red-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p class="font-semibold">Error displaying results</p>
                <p class="text-sm mt-2">${error.message}</p>
            </div>
        `;
    }
}

// Function to execute the edited generated SQL query
async function executeGeneratedQuery() {
    const sqlQuery = document.getElementById('generated-sql-query').value.trim();
    const connectionId = document.getElementById('agent-dataset-connection').value.trim();
    
    if (!sqlQuery) {
        showNotification('Please enter a SQL query to execute', 'warning');
        return;
    }
    
    if (!connectionId) {
        showNotification('Please select a database connection', 'warning');
        return;
    }
    
    try {
        // Update status to show loading
        document.getElementById('query-status').innerHTML = `
            <span class="text-blue-600"><i class="fas fa-spinner animate-spin mr-2"></i>Executing query...</span>
        `;
        
        // Execute the query
        const response = await fetch('/sql/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                connection_id: connectionId,
                sql_query: sqlQuery,
                limit: 100
            })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to execute query');
        }
        
        // Display results
        displayQueryResults(result);
        
        // Update status
        document.getElementById('query-status').innerHTML = `
            <span class="text-green-600"><i class="fas fa-check mr-2"></i>Query executed successfully</span>
        `;
        
        showNotification('Query executed successfully!', 'success');
        
    } catch (error) {
        console.error('Error executing query:', error);
        showNotification(error.message || 'Failed to execute query', 'error');
        
        // Show error in status
        document.getElementById('query-status').innerHTML = `
            <span class="text-red-600"><i class="fas fa-times mr-2"></i>Query execution failed</span>
        `;
        
        // Show error in results
        document.getElementById('query-results').innerHTML = `
            <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <div class="flex items-start">
                    <div class="flex-shrink-0">
                        <i class="fas fa-exclamation-circle text-red-400 text-xl"></i>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800 dark:text-red-300">Query Execution Failed</h3>
                        <div class="mt-2 text-sm text-red-700 dark:text-red-400">
                            <p>${error.message || 'Unknown error occurred'}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

// Function to reset the generated SQL query to its original state
function resetGeneratedQuery() {
    if (originalGeneratedSQL) {
        document.getElementById('generated-sql-query').value = originalGeneratedSQL;
        showNotification('SQL query reset to original', 'info');
    } else {
        showNotification('No original query to reset to', 'warning');
    }
}

// Function to save the edited SQL as a new dataset or update existing one
async function saveEditedDataset() {
    const name = document.getElementById('dataset-name').value.trim();
    const sqlQuery = document.getElementById('generated-sql-query').value.trim();
    const connectionId = document.getElementById('agent-dataset-connection').value.trim();
    
    if (!name || !sqlQuery || !connectionId) {
        showNotification('Please fill in all required fields', 'warning');
        return;
    }
    
    try {
        let response;
        let url;
        let requestData;
        
        // Check if we're editing an existing dataset
        if (isEditMode && currentEditingDataset) {
            // Update existing dataset using agent endpoint with dataset_id
            url = '/datasets/sql/agent-generate';
            requestData = {
                name: name, // Keep original name, don't add "(Edited)"
                question: document.getElementById('agent-question').value.trim() || 'Updated query',
                connection_id: connectionId,
                dataset_id: currentEditingDataset // This tells backend to update instead of create
            };
        } else {
            // Create new dataset
            url = '/datasets/sql';
            requestData = {
                name: name + ' (Edited)', // Only add "(Edited)" when creating new
                connection_id: connectionId,
                sql_query: sqlQuery
            };
        }
        
        response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to save dataset');
        }
        
        if (isEditMode && currentEditingDataset) {
            showNotification('Dataset updated successfully!', 'success');
            resetEditMode();
        } else {
            showNotification('Edited dataset saved successfully!', 'success');
        }
        
        await loadSqlDatasets();
        hideCreateDataset();
        
    } catch (error) {
        console.error('Error saving edited dataset:', error);
        showNotification(error.message || 'Failed to save edited dataset', 'error');
    }
}