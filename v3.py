<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Creation Tool - Streamlit Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #ffffff;
            color: #262730;
            line-height: 1.6;
        }

        .streamlit-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }

        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 300px;
            height: 100vh;
            background-color: #f0f2f6;
            padding: 1rem;
            overflow-y: auto;
            border-right: 1px solid #e6e9ef;
        }

        .main-content {
            margin-left: 320px;
            padding: 1rem;
        }

        .title {
            font-size: 2.5rem;
            font-weight: 600;
            color: #262730;
            margin-bottom: 1rem;
        }

        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #262730;
        }

        .file-uploader {
            margin-bottom: 1rem;
        }

        .file-uploader label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .file-input {
            width: 100%;
            padding: 0.5rem;
            border: 2px dashed #cccccc;
            border-radius: 0.5rem;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s;
        }

        .file-input:hover {
            border-color: #ff4b4b;
        }

        .file-uploaded {
            border-color: #00cc88;
            background-color: #f0fff4;
        }

        .selectbox {
            margin-bottom: 1rem;
        }

        .selectbox label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .selectbox select {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #d1d5db;
            border-radius: 0.375rem;
            font-size: 1rem;
        }

        .tabs {
            display: flex;
            border-bottom: 1px solid #e6e9ef;
            margin-bottom: 1rem;
        }

        .tab {
            padding: 0.75rem 1.5rem;
            border: none;
            background: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            font-size: 1rem;
            transition: all 0.3s;
        }

        .tab.active {
            border-bottom-color: #ff4b4b;
            color: #ff4b4b;
            font-weight: 600;
        }

        .tab:hover {
            background-color: #f8f9fa;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .expander {
            border: 1px solid #e6e9ef;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .expander-header {
            background-color: #f8f9fa;
            padding: 1rem;
            cursor: pointer;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .expander-content {
            padding: 1rem;
            display: none;
        }

        .expander.expanded .expander-content {
            display: block;
        }

        .expander-toggle {
            transition: transform 0.3s;
        }

        .expander.expanded .expander-toggle {
            transform: rotate(180deg);
        }

        .form-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .form-col {
            flex: 1;
        }

        .form-col label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .form-col select,
        .form-col input {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #d1d5db;
            border-radius: 0.375rem;
            font-size: 1rem;
        }

        .slider-container {
            margin-bottom: 1rem;
        }

        .slider-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .slider {
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: #e6e9ef;
            outline: none;
            -webkit-appearance: none;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ff4b4b;
            cursor: pointer;
        }

        .slider::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ff4b4b;
            cursor: pointer;
            border: none;
        }

        .button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.375rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s;
            margin-right: 1rem;
            margin-bottom: 1rem;
        }

        .button:hover {
            background-color: #e63946;
        }

        .button.secondary {
            background-color: #6c757d;
        }

        .button.secondary:hover {
            background-color: #5a6268;
        }

        .portfolio-card {
            border: 1px solid #e6e9ef;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .portfolio-header {
            background-color: #f8f9fa;
            padding: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .portfolio-content {
            padding: 1rem;
            display: none;
        }

        .portfolio-card.expanded .portfolio-content {
            display: block;
        }

        .alert {
            padding: 1rem;
            border-radius: 0.375rem;
            margin-bottom: 1rem;
        }

        .alert.info {
            background-color: #cce5ff;
            color: #0066cc;
            border: 1px solid #99ccff;
        }

        .alert.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .alert.warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .tracker-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .tracker-table th,
        .tracker-table td {
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid #e6e9ef;
        }

        .tracker-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.9rem;
        }

        .data-table th,
        .data-table td {
            padding: 0.5rem;
            text-align: left;
            border: 1px solid #e6e9ef;
        }

        .data-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }

        .subheader {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #262730;
        }

        .multiselect {
            border: 1px solid #d1d5db;
            border-radius: 0.375rem;
            padding: 0.5rem;
            min-height: 2.5rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
        }

        .multiselect-tag {
            background-color: #e9ecef;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .multiselect-tag .remove {
            cursor: pointer;
            color: #6c757d;
        }

        .divider {
            height: 1px;
            background-color: #e6e9ef;
            margin: 2rem 0;
        }

        .map-container {
            width: 100%;
            height: 500px;
            border: 1px solid #e6e9ef;
            border-radius: 0.5rem;
            margin: 1rem 0;
            overflow: hidden;
        }

        .map-legend {
            display: flex;
            gap: 2rem;
            margin-bottom: 1rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }

        .map-controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .map-stats {
            display: flex;
            gap: 2rem;
            margin-top: 1rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #ff4b4b;
        }

        .stat-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">Upload CSV files</div>
        
        <div class="file-uploader">
            <label>Customer Data</label>
            <div class="file-input file-uploaded" onclick="toggleFileUpload('customer')">
                <span id="customer-file">✓ customer_data.csv</span>
            </div>
        </div>
        
        <div class="file-uploader">
            <label>Banker Data</label>
            <div class="file-input file-uploaded" onclick="toggleFileUpload('banker')">
                <span id="banker-file">✓ banker_data.csv</span>
            </div>
        </div>
        
        <div class="file-uploader">
            <label>Branch Data</label>
            <div class="file-input file-uploaded" onclick="toggleFileUpload('branch')">
                <span id="branch-file">✓ branch_data.csv</span>
            </div>
        </div>
        
        <div class="sidebar-header" style="margin-top: 2rem;">Form Configuration</div>
        
        <div class="selectbox">
            <label>Number of Portfolios</label>
            <select id="num-forms" onchange="updateFormTabs()">
                <option value="1">1</option>
                <option value="2" selected>2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
            </select>
        </div>
        
        <div class="sidebar-header" style="margin-top: 2rem;">Live Portfolio Tracker</div>
        
        <table class="tracker-table">
            <thead>
                <tr>
                    <th>PortfolioID</th>
                    <th>Customers</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>PF001</td><td>25</td></tr>
                <tr><td>PF002</td><td>18</td></tr>
                <tr><td>PF003</td><td>22</td></tr>
            </tbody>
        </table>
        
        <div style="margin-top: 1rem;">
            <strong>Customer count per Form</strong>
            <div style="margin-top: 0.5rem;">
                <div>Form 1 → 35 Customer(s)</div>
                <div>Form 2 → 30 Customer(s)</div>
            </div>
        </div>
    </div>

    <div class="main-content">
        <h1 class="title">Portfolio creation tool</h1>
        
        <div class="selectbox" style="width: 300px; margin-bottom: 2rem;">
            <label>Select Page</label>
            <select>
                <option value="portfolio-assignment" selected>Portfolio Assignment</option>
                <option value="portfolio-mapping">Portfolio Mapping</option>
            </select>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab(1)">Form 1</button>
            <button class="tab" onclick="showTab(2)">Form 2</button>
        </div>

        <div class="tab-content active" id="tab-1">
            <div class="subheader">Form 1</div>
            
            <div class="expander expanded">
                <div class="expander-header" onclick="toggleExpander(this)">
                    <span>Select AU</span>
                    <span class="expander-toggle">▼</span>
                </div>
                <div class="expander-content">
                    <div class="form-row">
                        <div class="form-col">
                            <label>State (Form 1)</label>
                            <select>
                                <option selected>NY</option>
                                <option>CA</option>
                                <option>TX</option>
                                <option>FL</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>City (Form 1)</label>
                            <select>
                                <option selected>New York</option>
                                <option>Buffalo</option>
                                <option>Albany</option>
                                <option>Syracuse</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>AU (Form 1)</label>
                            <select>
                                <option selected>1001</option>
                                <option>1002</option>
                                <option>1003</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <div class="expander expanded">
                <div class="expander-header" onclick="toggleExpander(this)">
                    <span>Select Customers</span>
                    <span class="expander-toggle">▼</span>
                </div>
                <div class="expander-content">
                    <div class="form-row">
                        <div class="form-col">
                            <label>Role (Form 1)</label>
                            <select>
                                <option>All Roles</option>
                                <option selected>IN-MARKET</option>
                                <option>CENTRALIZED</option>
                                <option>Unassigned</option>
                                <option>Unmanaged</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>Customer State (Form 1)</label>
                            <select>
                                <option>All States</option>
                                <option selected>NY</option>
                                <option>NJ</option>
                                <option>CT</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>Portfolio Code (Form 1)</label>
                            <div class="multiselect">
                                <div class="multiselect-tag">
                                    PF001 <span class="remove">×</span>
                                </div>
                                <div class="multiselect-tag">
                                    PF002 <span class="remove">×</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="slider-container">
                                <label class="slider-label">Max Distance (Form 1): 60 km</label>
                                <input type="range" min="20" max="100" value="60" class="slider">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="slider-container">
                                <label class="slider-label">Minimum Revenue (Form 1): $5,000</label>
                                <input type="range" min="0" max="20000" value="5000" class="slider">
                            </div>
                        </div>
                    </div>
                    
                    <div class="slider-container">
                        <label class="slider-label">Minimum Deposit (Form 1): $100,000</label>
                        <input type="range" min="0" max="200000" value="100000" class="slider">
                    </div>
                </div>
            </div>

            <div class="alert warning">
                12 customers already assigned and removed
            </div>

            <div class="portfolio-card expanded">
                <div class="portfolio-header" onclick="togglePortfolio(this)">
                    <span>Portfolio PF001 - 45 customers</span>
                    <span>▼</span>
                </div>
                <div class="portfolio-content">
                    <div class="slider-container">
                        <label class="slider-label">Top N customers to select from Portfolio PF001</label>
                        <input type="range" min="0" max="45" value="25" class="slider">
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #6c757d;">25 customers selected</div>
                    </div>
                </div>
            </div>

            <div class="portfolio-card expanded">
                <div class="portfolio-header" onclick="togglePortfolio(this)">
                    <span>Portfolio PF002 - 32 customers</span>
                    <span>▼</span>
                </div>
                <div class="portfolio-content">
                    <div class="slider-container">
                        <label class="slider-label">Top N customers to select from Portfolio PF002</label>
                        <input type="range" min="0" max="32" value="18" class="slider">
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #6c757d;">18 customers selected</div>
                    </div>
                </div>
            </div>

            <button class="button" onclick="saveForm(1)">Save Form 1</button>
        </div>

        <div class="tab-content" id="tab-2">
            <div class="subheader">Form 2</div>
            
            <div class="expander expanded">
                <div class="expander-header" onclick="toggleExpander(this)">
                    <span>Select AU</span>
                    <span class="expander-toggle">▼</span>
                </div>
                <div class="expander-content">
                    <div class="form-row">
                        <div class="form-col">
                            <label>State (Form 2)</label>
                            <select>
                                <option>NY</option>
                                <option selected>CA</option>
                                <option>TX</option>
                                <option>FL</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>City (Form 2)</label>
                            <select>
                                <option selected>Los Angeles</option>
                                <option>San Francisco</option>
                                <option>San Diego</option>
                                <option>Sacramento</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>AU (Form 2)</label>
                            <select>
                                <option selected>2001</option>
                                <option>2002</option>
                                <option>2003</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <div class="expander expanded">
                <div class="expander-header" onclick="toggleExpander(this)">
                    <span>Select Customers</span>
                    <span class="expander-toggle">▼</span>
                </div>
                <div class="expander-content">
                    <div class="form-row">
                        <div class="form-col">
                            <label>Role (Form 2)</label>
                            <select>
                                <option>All Roles</option>
                                <option>IN-MARKET</option>
                                <option selected>CENTRALIZED</option>
                                <option>Unassigned</option>
                                <option>Unmanaged</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>Customer State (Form 2)</label>
                            <select>
                                <option>All States</option>
                                <option selected>CA</option>
                                <option>NV</option>
                                <option>AZ</option>
                            </select>
                        </div>
                        <div class="form-col">
                            <label>Portfolio Code (Form 2)</label>
                            <div class="multiselect">
                                <div class="multiselect-tag">
                                    PF003 <span class="remove">×</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="slider-container">
                                <label class="slider-label">Max Distance (Form 2): 75 km</label>
                                <input type="range" min="20" max="100" value="75" class="slider">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="slider-container">
                                <label class="slider-label">Minimum Revenue (Form 2): $7,500</label>
                                <input type="range" min="0" max="20000" value="7500" class="slider">
                            </div>
                        </div>
                    </div>
                    
                    <div class="slider-container">
                        <label class="slider-label">Minimum Deposit (Form 2): $150,000</label>
                        <input type="range" min="0" max="200000" value="150000" class="slider">
                    </div>
                </div>
            </div>

            <div class="portfolio-card expanded">
                <div class="portfolio-header" onclick="togglePortfolio(this)">
                    <span>Portfolio PF003 - 38 customers</span>
                    <span>▼</span>
                </div>
                <div class="portfolio-content">
                    <div class="slider-container">
                        <label class="slider-label">Top N customers to select from Portfolio PF003</label>
                        <input type="range" min="0" max="38" value="30" class="slider">
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #6c757d;">30 customers selected</div>
                    </div>
                </div>
            </div>

            <button class="button" onclick="saveForm(2)">Save Form 2</button>
        </div>

        <div class="divider"></div>

        <!-- Map Section -->
        <div class="subheader">Geographic Distribution</div>
        
        <div class="map-controls">
            <div class="form-col" style="width: 200px;">
                <label>Show Layer</label>
                <select id="map-layer" onchange="updateMapLayer()">
                    <option value="all">All Customers & AU</option>
                    <option value="form1">Form 1 Only</option>
                    <option value="form2">Form 2 Only</option>
                    <option value="au">AU Only</option>
                </select>
            </div>
            <div class="form-col" style="width: 200px;">
                <label>Distance Filter</label>
                <select id="distance-filter" onchange="updateDistanceFilter()">
                    <option value="all">All Distances</option>
                    <option value="30">Within 30km</option>
                    <option value="50">Within 50km</option>
                    <option value="75">Within 75km</option>
                </select>
            </div>
        </div>

        <div class="map-legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ff4b4b;"></div>
                <span>AU Locations</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #00cc88;"></div>
                <span>Form 1 Customers</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #0066cc;"></div>
                <span>Form 2 Customers</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffa500;"></div>
                <span>Unassigned Customers</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #cccccc; border: 1px solid #999;"></div>
                <span>Distance Circle</span>
            </div>
        </div>

        <div class="map-container">
            <div id="map" style="width: 100%; height: 100%;"></div>
        </div>

        <div class="map-stats">
            <div class="stat-item">
                <div class="stat-value">2</div>
                <div class="stat-label">AU Locations</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">43</div>
                <div class="stat-label">Form 1 Customers</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">30</div>
                <div class="stat-label">Form 2 Customers</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">52.3 km</div>
                <div class="stat-label">Avg Distance</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">127</div>
                <div class="stat-label">Total Customers</div>
            </div>
        </div>

        <div class="divider"></div>

        <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
            <button class="button" onclick="showRecommendations()">Recommended form</button>
            <button class="button secondary" onclick="saveAllForms()">Save all forms</button>
        </div>

        <div class="alert success" id="save-message" style="display: none;">
            All Forms are saved successfully
        </div>

        <div id="recommendations" style="display: none;">
            <div class="subheader">Form reassignment</div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Customer ID</th>
                        <th>Current Form</th>
                        <th>Current Distance</th>
                        <th>Recommended Form</th>
                        <th>Recommended Distance</th>
                        <th>Savings (km)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>CUST001</td>
                        <td>Form 1</td>
                        <td>45.2</td>
                        <td>Form 2</td>
                        <td>32.8</td>
                        <td>12.4</td>
                    </tr>
                    <tr>
                        <td>CUST002</td>
                        <td>Form 2</td>
                        <td>58.7</td>
                        <td>Form 1</td>
                        <td>41.3</td>
                        <td>17.4</td>
                    </tr>
                    <tr>
                        <td>CUST003</td>
                        <td>Form 1</td>
                        <td>62.1</td>
                        <td>Form 2</td>
                        <td>38.9</td>
                        <td>23.2</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div id="final-results" style="display: none;">
            <div class="subheader">Final Results</div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Customer ID</th>
                        <th>Form ID</th>
                        <th>Portfolio Code</th>
                        <th>Distance (km)</th>
                        <th>Revenue ($)</th>
                        <th>Deposit ($)</th>
                        <th>State</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>CUST001</td>
                        <td>1</td>
                        <td>PF001</td>
                        <td>32.8</td>
                        <td>12,500</td>
                        <td>125,000</td>
                        <td>NY</td>
                    </tr>
                    <tr>
                        <td>CUST002</td>
                        <td>1</td>
                        <td>PF001</td>
                        <td>41.3</td>
                        <td>8,750</td>
                        <td>110,000</td>
                        <td>NY</td>
                    </tr>
                    <tr>
                        <td>CUST003</td>
                        <td>2</td>
                        <td>PF003</td>
                        <td>38.9</td>
                        <td>15,200</td>
                        <td>165,000</td>
                        <td>CA</td>
                    </tr>
                    <tr>
                        <td>CUST004</td>
                        <td>2</td>
                        <td>PF003</td>
                        <td>47.2</td>
                        <td>9,800</td>
                        <td>155,000</td>
                        <td>CA</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Include Leaflet CSS and JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
        let map;
        let auMarkers = [];
        let customerMarkers = [];
        let distanceCircles = [];

        // Sample data - in real implementation, this would come from your backend
        const auData = [
            { id: 1001, name: "AU New York", lat: 40.7128, lng: -74.0060, city: "New York", state: "NY" },
            { id: 2001, name: "AU Los Angeles", lat: 34.0522, lng: -118.2437, city: "Los Angeles", state: "CA" }
        ];

        const customerData = [
            { id: "CUST001", name: "Customer 1", lat: 40.7580, lng: -73.9855, form: 1, distance: 32.8, revenue: 12500, deposit: 125000 },
            { id: "CUST002", name: "Customer 2", lat: 40.6892, lng: -74.0445, form: 1, distance: 41.3, revenue: 8750, deposit: 110000 },
            { id: "CUST003", name: "Customer 3", lat: 40.7505, lng: -73.9934, form: 1, distance: 28.5, revenue: 15200, deposit: 165000 },
            { id: "CUST004", name: "Customer 4", lat: 34.0928, lng: -118.3287, form: 2, distance: 47.2, revenue: 9800, deposit: 155000 },
            { id: "CUST005", name: "Customer 5", lat: 34.0194, lng: -118.2108, form: 2, distance: 38.9, revenue: 11200, deposit: 142000 },
            { id: "CUST006", name: "Customer 6", lat: 34.0736, lng: -118.4004, form: 2, distance: 52.1, revenue: 7500, deposit: 118000 },
            { id: "CUST007", name: "Customer 7", lat: 40.7282, lng: -73.7949, form: null, distance: 65.2, revenue: 6200, deposit: 95000 },
            { id: "CUST008", name: "Customer 8", lat: 34.1478, lng: -118.1445, form: null, distance: 71.8, revenue: 5800, deposit: 88000 }
        ];

        function initializeMap() {
            // Initialize the map
            map = L.map('map').setView([39.8283, -98.5795], 4); // Center of US

            // Add tile layer
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: ' OpenStreetMap contributors'
            }).addTo(map);

            // Add AU markers
            auData.forEach(au => {
                const marker = L.marker([au.lat, au.lng], {
                    icon: L.divIcon({
                        html: `<div style="background-color: #ff4b4b; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
                        className: 'au-marker',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10]
                    })
                }).addTo(map);

                marker.bindPopup(`
                    <div style="font-size: 14px;">
                        <strong>${au.name}</strong><br>
                        ID: ${au.id}<br>
                        Location: ${au.city}, ${au.state}<br>
                        Coordinates: ${au.lat.toFixed(4)}, ${au.lng.toFixed(4)}
                    </div>
                `);

                auMarkers.push(marker);

                // Add distance circle (60km default)
                const circle = L.circle([au.lat, au.lng], {
                    radius: 60000, // 60km in meters
                    color: '#cccccc',
                    fillColor: '#cccccc',
                    fillOpacity: 0.1,
                    weight: 2,
                    dashArray: '5, 5'
                }).addTo(map);

                distanceCircles.push(circle);
            });

            // Add customer markers
            customerData.forEach(customer => {
                let color = '#ffa500'; // Default orange for unassigned
                if (customer.form === 1) color = '#00cc88'; // Green for Form 1
                if (customer.form === 2) color = '#0066cc'; // Blue for Form 2

                const marker = L.marker([customer.lat, customer.lng], {
                    icon: L.divIcon({
                        html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 1px 2px rgba(0,0,0,0.3);"></div>`,
                        className: 'customer-marker',
                        iconSize: [12, 12],
                        iconAnchor: [6, 6]
                    })
                }).addTo(map);

                marker.bindPopup(`
                    <div style="font-size: 12px;">
                        <strong>${customer.name}</strong><br>
                        ID: ${customer.id}<br>
                        Form: ${customer.form || 'Unassigned'}<br>
                        Distance: ${customer.distance} km<br>
                        Revenue: ${customer.revenue.toLocaleString()}<br>
                        Deposit: ${customer.deposit.toLocaleString()}<br>
                        Coordinates: ${customer.lat.toFixed(4)}, ${customer.lng.toFixed(4)}
                    </div>
                `);

                customerMarkers.push({ marker, customer });
            });

            // Fit map to show all markers
            const allMarkers = [...auMarkers, ...customerMarkers.map(c => c.marker)];
            if (allMarkers.length > 0) {
                const group = new L.featureGroup(allMarkers);
                map.fitBounds(group.getBounds().pad(0.1));
            }
        }

        function updateMapLayer() {
            const layer = document.getElementById('map-layer').value;
            
            // Show/hide customer markers based on selection
            customerMarkers.forEach(({ marker, customer }) => {
                if (layer === 'all' || 
                    (layer === 'form1' && customer.form === 1) ||
                    (layer === 'form2' && customer.form === 2) ||
                    (layer === 'au' && false)) { // AU layer shows no customers
                    marker.addTo(map);
                } else {
                    map.removeLayer(marker);
                }
            });

            // Always show AU markers unless specifically filtering
            auMarkers.forEach(marker => {
                marker.addTo(map);
            });
        }

        function updateDistanceFilter() {
            const maxDistance = document.getElementById('distance-filter').value;
            
            if (maxDistance === 'all') {
                // Show all customers
                customerMarkers.forEach(({ marker }) => {
                    marker.addTo(map);
                });
            } else {
                // Filter by distance
                const maxDist = parseInt(maxDistance);
                customerMarkers.forEach(({ marker, customer }) => {
                    if (customer.distance <= maxDist) {
                        marker.addTo(map);
                    } else {
                        map.removeLayer(marker);
                    }
                });
            }

            // Update distance circles
            distanceCircles.forEach(circle => {
                if (maxDistance === 'all') {
                    circle.setRadius(60000); // Default 60km
                } else {
                    circle.setRadius(parseInt(maxDistance) * 1000); // Convert km to meters
                }
            });
        }

        function toggleExpander(header) {
            const expander = header.parentElement;
            expander.classList.toggle('expanded');
        }

        function togglePortfolio(header) {
            const portfolio = header.parentElement;
            portfolio.classList.toggle('expanded');
        }

        function showTab(tabNumber) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(`tab-${tabNumber}`).classList.add('active');
            document.querySelectorAll('.tab')[tabNumber - 1].classList.add('active');
        }

        function updateFormTabs() {
            const numForms = document.getElementById('num-forms').value;
            const tabsContainer = document.querySelector('.tabs');
            
            // Clear existing tabs
            tabsContainer.innerHTML = '';
            
            // Create new tabs
            for (let i = 1; i <= numForms; i++) {
                const tab = document.createElement('button');
                tab.className = 'tab';
                if (i === 1) tab.classList.add('active');
                tab.textContent = `Form ${i}`;
                tab.onclick = () => showTab(i);
                tabsContainer.appendChild(tab);
            }
        }

        function saveForm(formId) {
            const alert = document.createElement('div');
            alert.className = 'alert success';
            alert.textContent = `Form ${formId} saved with 43 customers`;
            
            const tabContent = document.getElementById(`tab-${formId}`);
            tabContent.insertBefore(alert, tabContent.querySelector('.portfolio-card'));
            
            // Remove alert after 3 seconds
            setTimeout(() => {
                alert.remove();
            }, 3000);
        }

        function showRecommendations() {
            document.getElementById('recommendations').style.display = 'block';
        }

        function saveAllForms() {
            document.getElementById('save-message').style.display = 'block';
            document.getElementById('final-results').style.display = 'block';
        }

        function toggleFileUpload(type) {
            const fileSpan = document.getElementById(`${type}-file`);
            if (fileSpan.textContent.includes('✓')) {
                fileSpan.textContent = `Choose ${type}_data.csv file`;
                fileSpan.parentElement.classList.remove('file-uploaded');
            } else {
                fileSpan.textContent = `✓ ${type}_data.csv`;
                fileSpan.parentElement.classList.add('file-uploaded');
            }
        }

        // Initialize the map when the page loads
        window.addEventListener('load', function() {
            setTimeout(initializeMap, 100); // Small delay to ensure DOM is ready
        });

        // Initialize sliders with value display
        document.addEventListener('DOMContentLoaded', function() {
            const sliders = document.querySelectorAll('.slider');
            sliders.forEach(slider => {
                slider.addEventListener('input', function() {
                    const label = this.parentElement.querySelector('.slider-label');
                    const value = this.value;
                    const text = label.textContent;
                    const colonIndex = text.indexOf(':');
                    if (colonIndex !== -1) {
                        const prefix = text.substring(0, colonIndex + 1);
                        if (text.includes('Distance')) {
                            label.textContent = `${prefix} ${value} km`;
                        } else if (text.includes('Revenue')) {
                            label.textContent = `${prefix} ${parseInt(value).toLocaleString()}`;
                        } else if (text.includes('Deposit')) {
                            label.textContent = `${prefix} ${parseInt(value).toLocaleString()}`;
                        } else {
                            label.textContent = `${prefix} ${value}`;
                        }
                    }
                });
            });
        });
