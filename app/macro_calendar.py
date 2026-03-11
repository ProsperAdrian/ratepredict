"""
Macro Calendar: Recurring events that affect the USD/NGN exchange rate.

Direction convention:
  UP   = USD/NGN rises = Naira weakens
  DOWN = USD/NGN falls = Naira strengthens
  MIXED = depends on outcome

Magnitude: LARGE / MEDIUM / SMALL / NEGLIGIBLE
"""

from __future__ import annotations

import calendar as _cal
from collections import defaultdict


EVENTS: list[dict] = [
    # =====================================================================
    # NIGERIA DOMESTIC
    # =====================================================================
    {
        "category": "Nigeria",
        "event": "CBN Monetary Policy Committee (MPC)",
        "frequency": "6x/year (bi-monthly)",
        "timing": "Jan, Mar, May, Jul, Sep, Nov — Tuesday afternoon WAT",
        "impact_window": "3-5 days before, peaks on decision day, lingers 2-3 days after",
        "direction": "Hike → DOWN, Hold (when hike expected) → UP, Cut → UP",
        "magnitude": "LARGE",
        "mechanism": "Rate changes alter yield differential vs USD. Higher rates attract carry trade inflows, tighten Naira liquidity, reduce speculative dollar demand.",
    },
    {
        "category": "Nigeria",
        "event": "CBN FX Interventions / SMIS Auctions",
        "frequency": "Weekly (Mon-Thu)",
        "timing": "Wholesale SMIS Mon/Tue, Retail SMIS Wed/Thu",
        "impact_window": "Same day",
        "direction": "Large auction → DOWN, Reduced/suspended → UP",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Direct USD supply into the market reduces scarcity premium. Authorized dealers fulfill customer orders, reducing parallel market demand.",
    },
    {
        "category": "Nigeria",
        "event": "Nigerian CPI / Inflation (NBS)",
        "frequency": "Monthly",
        "timing": "~15th of the following month",
        "impact_window": "Same day through next MPC meeting",
        "direction": "High CPI → UP (purchasing power erosion), but can → DOWN if market expects rate hike response",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Key input for MPC decisions. High inflation erodes real returns on Naira assets.",
    },
    {
        "category": "Nigeria",
        "event": "FAAC Monthly Distribution",
        "frequency": "Monthly",
        "timing": "~20th-25th of each month",
        "impact_window": "2-3 days after meeting (when states receive funds)",
        "direction": "UP — injects Naira liquidity that leaks into FX demand",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "States spend allocations rapidly on salaries and contracts. Liquidity injection means more Naira chasing dollars.",
    },
    {
        "category": "Nigeria",
        "event": "Company Income Tax (CIT) Deadline",
        "frequency": "Annual",
        "timing": "Provisional: Mar 31, Final: Jun 30 (for Dec year-end companies)",
        "impact_window": "1-2 weeks before deadline",
        "direction": "DOWN — multinationals convert USD to NGN for tax payments",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Foreign-owned companies must convert FX to Naira to pay FIRS. Creates temporary USD supply.",
    },
    {
        "category": "Nigeria",
        "event": "Petroleum Profit Tax (PPT) Payments",
        "frequency": "Monthly installments, annual reconciliation Q1",
        "timing": "Last working day of each month; large Q1 reconciliation",
        "impact_window": "1-2 days before payment deadline",
        "direction": "DOWN — IOCs convert USD to NGN",
        "magnitude": "MEDIUM",
        "mechanism": "International oil companies (Shell, Total, Eni) convert significant USD to NGN for tax payments.",
    },
    {
        "category": "Nigeria",
        "event": "CBN External Reserves Report",
        "frequency": "Weekly",
        "timing": "Updated Thursday/Friday on CBN website",
        "impact_window": "Same day",
        "direction": "Rising reserves → DOWN, Falling reserves → UP",
        "magnitude": "MEDIUM",
        "mechanism": "Reserves indicate CBN's capacity to defend the Naira. Depletion signals vulnerability and triggers speculative attacks.",
    },
    {
        "category": "Nigeria",
        "event": "Nigerian Oil Production Data (NNPC/OPEC)",
        "frequency": "Monthly",
        "timing": "Mid-month, ~2 month lag",
        "impact_window": "Same day",
        "direction": "Rising production → DOWN, Falling (vandalism/theft) → UP",
        "magnitude": "MEDIUM",
        "mechanism": "Production volume determines how much of the oil price Nigeria captures in export revenue.",
    },
    {
        "category": "Nigeria",
        "event": "FGN Bond Auctions (DMO)",
        "frequency": "Monthly",
        "timing": "3rd or 4th Wednesday of the month",
        "impact_window": "2-3 days before through settlement (T+2)",
        "direction": "Strong foreign demand → DOWN (FX inflows), Weak auction → UP",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Foreign investors convert USD to NGN to buy bonds. Attractive yields pull money into Naira assets.",
    },
    {
        "category": "Nigeria",
        "event": "NTB / OMO Auctions",
        "frequency": "Bi-weekly (NTB), irregular (OMO)",
        "timing": "NTB: 1st/2nd Wednesday; OMO: Tue/Thu at CBN discretion",
        "impact_window": "Same day through settlement",
        "direction": "High rates → DOWN (yield attraction), Large OMO mop-up → DOWN (drains Naira liquidity)",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Higher NTB rates increase the opportunity cost of holding dollars. OMO sales absorb excess Naira liquidity.",
    },
    {
        "category": "Nigeria",
        "event": "Eurobond Coupon / Principal Payments",
        "frequency": "Semi-annual (coupons); maturities on specific dates",
        "timing": "Varies by tranche — typically Feb, Jul, Nov",
        "impact_window": "1-2 weeks before payment",
        "direction": "UP — government demands USD for coupon payments, drawing on reserves",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "FGN must pay Eurobond coupons in USD, reducing external reserves and diverting USD supply.",
    },
    {
        "category": "Nigeria",
        "event": "Government Budget Presentation & Signing",
        "frequency": "Annual",
        "timing": "Presentation: Oct-Dec, Passage: Dec-Mar",
        "impact_window": "1-2 days around presentation and signing",
        "direction": "Expansionary/high deficit → UP, Credible fiscal plan → DOWN",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Large deficits financed by CBN overdraft are inflationary and weaken the Naira.",
    },
    {
        "category": "Nigeria",
        "event": "Nigerian Elections",
        "frequency": "Every 4 years (next: Feb/Mar 2027)",
        "timing": "Presidential: February, Governorship: March",
        "impact_window": "6-9 months before through inauguration (May 29)",
        "direction": "UP — massive and prolonged pressure",
        "magnitude": "VERY LARGE",
        "mechanism": "Politicians hoard USD, political spending floods economy with Naira, capital flight, CBN leadership uncertainty.",
    },
    {
        "category": "Nigeria",
        "event": "Fuel Subsidy Removal / PMS Price Adjustments",
        "frequency": "Irregular but recurring",
        "timing": "Often last/first day of month",
        "impact_window": "Immediate, persists 1-4 weeks",
        "direction": "Initially UP (panic, inflation fears), medium-term can → DOWN (fiscal savings)",
        "magnitude": "LARGE",
        "mechanism": "Price increases trigger inflation expectations. But subsidy removal saves $5-10B annually, improving fiscal position.",
    },
    {
        "category": "Nigeria",
        "event": "NLNG Dividend Repatriation",
        "frequency": "Annual (sometimes semi-annual)",
        "timing": "Q2 (after annual results)",
        "impact_window": "Announcement and payment date",
        "direction": "UP — foreign shareholders repatriate dividends in USD",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Foreign JV partners (Shell, Total, Eni) repatriate their share of NLNG dividends ($1-3B annually).",
    },
    {
        "category": "Nigeria",
        "event": "Dangote Refinery Milestones",
        "frequency": "Irregular (ramp-up phase 2024-2026)",
        "timing": "Production milestone announcements",
        "impact_window": "Same day to weeks after",
        "direction": "Successful ramp-up → DOWN, Delays/shutdowns → UP",
        "magnitude": "LARGE (structural)",
        "mechanism": "Replaces $15-20B/year in refined petroleum imports, fundamentally reducing structural FX demand.",
    },
    # =====================================================================
    # SEASONAL / CALENDAR (NIGERIA-FOCUSED)
    # =====================================================================
    {
        "category": "Seasonal",
        "event": "Christmas / New Year Demand Surge",
        "frequency": "Annual",
        "timing": "Builds from late Oct, peaks Dec 10-24",
        "impact_window": "Late October through mid-January",
        "direction": "UP — one of the strongest seasonal patterns",
        "magnitude": "LARGE",
        "mechanism": "Import surge (retail goods), diaspora travel, foreign tuition for Jan term, year-end bonuses, BTA/PTA demand.",
    },
    {
        "category": "Seasonal",
        "event": "Foreign University Tuition Payments (Sep Intake)",
        "frequency": "Annual",
        "timing": "Peak demand: July-August",
        "impact_window": "June through September",
        "direction": "UP — 100K+ students needing USD/GBP/CAD",
        "magnitude": "LARGE",
        "mechanism": "Parents/students seek $1-2B+ in FX for tuition, accommodation, living expenses. Much flows through parallel market.",
    },
    {
        "category": "Seasonal",
        "event": "Foreign University Tuition (Jan Intake)",
        "frequency": "Annual",
        "timing": "Peak demand: November-January",
        "impact_window": "November through January",
        "direction": "UP — compounds with Christmas demand",
        "magnitude": "MEDIUM",
        "mechanism": "Smaller cohort than September but overlaps with Christmas demand surge.",
    },
    {
        "category": "Seasonal",
        "event": "Hajj Season FX Demand",
        "frequency": "Annual (Islamic calendar, shifts ~11 days earlier)",
        "timing": "2-3 months before Eid al-Adha (~May-Jun 2026)",
        "impact_window": "8-12 weeks before Hajj departure, peaks 2-4 weeks before",
        "direction": "UP — 70-100K pilgrims needing $3-5K each = $300-500M",
        "magnitude": "MEDIUM",
        "mechanism": "NAHCON bulk-purchases USD for pilgrim expenses. CBN may provide special allocation but shortfall spills to parallel market.",
    },
    {
        "category": "Seasonal",
        "event": "Ramadan",
        "frequency": "Annual (Islamic calendar)",
        "timing": "~30 days, ~Feb-Mar 2026",
        "impact_window": "1-2 weeks before through the month",
        "direction": "UP — increased food import demand",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Seasonal spike in imported foodstuffs (rice, sugar, dates). Trading volumes may decrease during Ramadan.",
    },
    {
        "category": "Seasonal",
        "event": "Eid al-Fitr / Eid al-Adha",
        "frequency": "Twice yearly",
        "timing": "End of Ramadan; ~70 days after Eid al-Fitr",
        "impact_window": "1 week before each",
        "direction": "UP — consumer spending surge",
        "magnitude": "SMALL",
        "mechanism": "Festival spending, gift-giving, travel, livestock imports for Eid al-Adha.",
    },
    {
        "category": "Seasonal",
        "event": "Diaspora Remittance Peaks",
        "frequency": "Annual cycle",
        "timing": "Peaks: Nov-Dec (Christmas), Mar-Apr (Easter), Aug-Sep (back-to-school)",
        "impact_window": "2-3 weeks before holidays",
        "direction": "DOWN — remittances provide USD supply ($20-25B annually)",
        "magnitude": "MEDIUM",
        "mechanism": "Diaspora Nigerians send USD home. Partially offsets demand surges during same periods.",
    },
    {
        "category": "Seasonal",
        "event": "Q4 Dollar Demand Surge",
        "frequency": "Annual",
        "timing": "October through December, peaking Nov-Dec",
        "impact_window": "Early October through year-end",
        "direction": "UP — worst quarter for Naira historically",
        "magnitude": "LARGE",
        "mechanism": "Christmas imports + tuition + repatriation + budget cycle + Hajj spillover + emigration transfers all compound.",
    },
    {
        "category": "Seasonal",
        "event": "January Effect",
        "frequency": "Annual",
        "timing": "First 2-3 weeks of January",
        "impact_window": "First trading day through mid-January",
        "direction": "UP — new year demand restart",
        "magnitude": "MEDIUM",
        "mechanism": "Businesses restock imports, new budget spending kicks in, oil earnings from December holidays lag.",
    },
    {
        "category": "Seasonal",
        "event": "End-of-Quarter Window Dressing",
        "frequency": "Quarterly",
        "timing": "Last 5-7 trading days of Mar, Jun, Sep, Dec",
        "impact_window": "Last week of quarter",
        "direction": "MIXED — increased volatility, banks may sell/buy USD for reporting",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Banks window-dress balance sheets. Multinational corporates repatriate quarterly earnings.",
    },
    {
        "category": "Seasonal",
        "event": "Planting Season / Agricultural Imports",
        "frequency": "Annual",
        "timing": "March-June",
        "impact_window": "March-April peak (fertilizer, chemicals imports)",
        "direction": "UP — import demand for agricultural inputs",
        "magnitude": "SMALL",
        "mechanism": "Demand for imported fertilizers, chemicals, farm equipment creates modest FX demand.",
    },
    {
        "category": "Seasonal",
        "event": "Cocoa / Agricultural Export Season",
        "frequency": "Annual",
        "timing": "Main crop: Oct-Jan, Light crop: May-Aug",
        "impact_window": "During harvest and export months",
        "direction": "DOWN — FX inflows from exports",
        "magnitude": "SMALL",
        "mechanism": "Export proceeds repatriated and converted to Naira. Nigeria is 4th largest cocoa producer.",
    },
    # =====================================================================
    # US / FEDERAL RESERVE
    # =====================================================================
    {
        "category": "US/Fed",
        "event": "FOMC Rate Decision",
        "frequency": "8x/year",
        "timing": "Jan, Mar, May, Jun, Jul, Sep, Nov, Dec — Wed 2pm ET (8pm WAT)",
        "impact_window": "3-5 days before, peaks on decision day, 24-48h digestion",
        "direction": "Hike/hawkish → UP, Cut/dovish → DOWN",
        "magnitude": "LARGE",
        "mechanism": "US rate differentials drive global capital allocation. Higher US rates increase opportunity cost of holding NGN assets, triggering EM outflows.",
    },
    {
        "category": "US/Fed",
        "event": "US CPI (Consumer Price Index)",
        "frequency": "Monthly",
        "timing": "~10th-13th of month, 8:30am ET",
        "impact_window": "Same day, sharp reaction in first 5 minutes",
        "direction": "Hot CPI → UP, Cool CPI → DOWN",
        "magnitude": "LARGE",
        "mechanism": "Inflation is the Fed's primary constraint. Higher inflation forces restrictive policy, strengthening dollar through rate differential.",
    },
    {
        "category": "US/Fed",
        "event": "US Non-Farm Payrolls (NFP)",
        "frequency": "Monthly",
        "timing": "First Friday of month, 8:30am ET",
        "impact_window": "Same day, pre-positioning starts Thursday evening",
        "direction": "Strong jobs → UP, Weak jobs → DOWN",
        "magnitude": "LARGE",
        "mechanism": "Employment data is the Fed's dual-mandate fulcrum. Strong jobs = less urgency to ease = higher-for-longer rates.",
    },
    {
        "category": "US/Fed",
        "event": "Core PCE Price Index (Fed's preferred measure)",
        "frequency": "Monthly",
        "timing": "Last Thu/Fri of month, 8:30am ET",
        "impact_window": "Same day",
        "direction": "Hot → UP, Cool → DOWN",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Fed explicitly targets 2% Core PCE. When it diverges from CPI, it directly recalibrates rate expectations.",
    },
    {
        "category": "US/Fed",
        "event": "Jackson Hole Symposium",
        "frequency": "Annual",
        "timing": "Last weekend of August — Chair speech Friday ~10am ET",
        "impact_window": "Speculation 1-2 weeks before, impact persists for weeks",
        "direction": "Hawkish speech → UP, Dovish speech → DOWN",
        "magnitude": "LARGE",
        "mechanism": "Fed Chair previews policy trajectory. Sets narrative for the next quarter.",
    },
    {
        "category": "US/Fed",
        "event": "Fed Chair Congressional Testimony",
        "frequency": "Twice yearly (Feb/Mar and Jun/Jul)",
        "timing": "Morning ~10am ET, over two days (House then Senate)",
        "impact_window": "Same day, Q&A matters more than prepared remarks",
        "direction": "Hawkish → UP, Dovish → DOWN",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Real-time Q&A reveals policy thinking beyond prepared statements.",
    },
    {
        "category": "US/Fed",
        "event": "FOMC Minutes",
        "frequency": "8x/year",
        "timing": "3 weeks after each FOMC, Wed 2pm ET",
        "impact_window": "Same day, 24h window",
        "direction": "Hawkish tone → UP, Dovish tone → DOWN",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Reveals internal debates that can shift expectations for future meetings.",
    },
    {
        "category": "US/Fed",
        "event": "US GDP (Advance Estimate)",
        "frequency": "Quarterly",
        "timing": "~4 weeks after quarter end, 8:30am ET",
        "impact_window": "Same day",
        "direction": "Strong → UP, Weak → complex (initially UP safe-haven, then DOWN on rate cut bets)",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Broadest measure of economic health. Confirms or contradicts the monthly indicator narrative.",
    },
    {
        "category": "US/Fed",
        "event": "ISM Services PMI",
        "frequency": "Monthly",
        "timing": "3rd business day of month, 10am ET",
        "impact_window": "Same day, 1-3 hours",
        "direction": "Above 50/consensus → UP, Below → DOWN",
        "magnitude": "MEDIUM",
        "mechanism": "Services = 80% of US economy. Prices-paid subcomponent is a key inflation indicator.",
    },
    {
        "category": "US/Fed",
        "event": "US Retail Sales",
        "frequency": "Monthly",
        "timing": "~14th-17th of month, 8:30am ET",
        "impact_window": "Same day, 1-3 hours",
        "direction": "Strong → UP, Weak → DOWN",
        "magnitude": "MEDIUM",
        "mechanism": "Consumer spending = ~70% of US GDP. Signals whether economy is on track for soft/hard landing.",
    },
    {
        "category": "US/Fed",
        "event": "US Treasury Quarterly Refunding (QRA)",
        "frequency": "Quarterly",
        "timing": "Last Wed of Jan, Apr, Jul, Oct",
        "impact_window": "Announcement day through auction week",
        "direction": "Larger-than-expected issuance → UP (higher yields attract capital to USD)",
        "magnitude": "MEDIUM",
        "mechanism": "Treasury supply dynamics affect long-term rates and term premium. Higher yields drain liquidity from EM.",
    },
    {
        "category": "US/Fed",
        "event": "US Debt Ceiling Events",
        "frequency": "Irregular (~every 1-3 years)",
        "timing": "Peaks in weeks before X-date",
        "impact_window": "Slow build over weeks, acute in final 1-2 weeks, TGA rebuild 4-8 weeks post-resolution",
        "direction": "During standoff → UP (paradoxically), Post-resolution TGA rebuild → UP (drains dollar liquidity)",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Resolution TGA rebuild via massive T-bill issuance sucks dollar liquidity out of global markets, tightening EM conditions.",
    },
    {
        "category": "US/Fed",
        "event": "US Presidential / Midterm Elections",
        "frequency": "Every 2 years (Nov)",
        "timing": "First Tuesday after first Monday in November",
        "impact_window": "3-6 months before through weeks after",
        "direction": "Pre-election uncertainty → UP, Republican/tariff risk → UP, Divided government → DOWN",
        "magnitude": "LARGE (presidential), MEDIUM (midterms)",
        "mechanism": "Trade/tariff policy affects EM flows. Fiscal policy affects deficit/rates. Election premium builds from September.",
    },
    {
        "category": "US/Fed",
        "event": "Year-End Dollar Funding Squeeze",
        "frequency": "Annual",
        "timing": "Mid-November through first week of January",
        "impact_window": "Builds from mid-Nov, peaks Dec 31 cross-year turn",
        "direction": "UP — one of the most reliable seasonal patterns",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Global banks reduce balance sheets for year-end reporting, cutting dollar lending to EM. Cross-currency basis widens.",
    },
    {
        "category": "US/Fed",
        "event": "US Tax Season",
        "frequency": "Annual",
        "timing": "Late March through April 15",
        "impact_window": "Gradual flow over 3 weeks",
        "direction": "DOWN during late Mar to mid-Apr (repatriation improves EM dollar liquidity), then mildly UP after",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "US multinationals draw down offshore dollar holdings for tax payments, temporarily improving EM liquidity.",
    },
    {
        "category": "US/Fed",
        "event": "Employment Cost Index (ECI)",
        "frequency": "Quarterly",
        "timing": "Last Thu/Fri of Jan, Apr, Jul, Oct",
        "impact_window": "Same day",
        "direction": "High → UP (wage-price spiral risk), Low → DOWN",
        "magnitude": "MEDIUM",
        "mechanism": "Fed's preferred wage measure. Controls for compositional shifts unlike average hourly earnings.",
    },
    # =====================================================================
    # OIL & ENERGY
    # =====================================================================
    {
        "category": "Oil/Energy",
        "event": "OPEC/OPEC+ Full Ministerial Meetings",
        "frequency": "2x/year (Jun, Dec) + extraordinary sessions",
        "timing": "Early-mid June; late Nov/early Dec",
        "impact_window": "2-5 days before (leaks), peak on decision day, 1-2 days after",
        "direction": "Production cuts → DOWN (oil up), Production increases → UP (oil down)",
        "magnitude": "LARGE",
        "mechanism": "Oil = ~90% of Nigeria's FX earnings. Brent price directly determines dollar supply in Nigerian economy.",
    },
    {
        "category": "Oil/Energy",
        "event": "OPEC+ JMMC Monitoring Committee",
        "frequency": "Every 2 months",
        "timing": "First week of Feb, Apr, Jun, Aug, Oct, Dec",
        "impact_window": "1-2 days before through 1 day after",
        "direction": "Same as OPEC — compliance/quota signals",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Monitors compliance. Non-compliance flooding market weakens oil prices.",
    },
    {
        "category": "Oil/Energy",
        "event": "US EIA Weekly Petroleum Report",
        "frequency": "Weekly",
        "timing": "Wednesday 10:30am ET",
        "impact_window": "Same day, within minutes",
        "direction": "Inventory build → UP (oil down), Inventory draw → DOWN (oil up)",
        "magnitude": "SMALL",
        "mechanism": "Signals US supply/demand balance. Large unexpected draws indicate tight global supply.",
    },
    {
        "category": "Oil/Energy",
        "event": "IEA Monthly Oil Market Report",
        "frequency": "Monthly",
        "timing": "~12th-15th of month",
        "impact_window": "Same day",
        "direction": "Demand upgrade → DOWN (oil up), Demand downgrade → UP (oil down)",
        "magnitude": "MEDIUM",
        "mechanism": "IEA demand forecasts are the benchmark for global oil demand outlook. Revisions shift Brent pricing.",
    },
    {
        "category": "Oil/Energy",
        "event": "OPEC Monthly Oil Market Report (MOMR)",
        "frequency": "Monthly",
        "timing": "~11th-13th of month",
        "impact_window": "Same day",
        "direction": "Same as IEA — demand/supply forecast revisions",
        "magnitude": "MEDIUM",
        "mechanism": "Contains actual production data revealing compliance vs quotas. Nigeria's own production figures included.",
    },
    {
        "category": "Oil/Energy",
        "event": "Brent Crude Seasonal Pattern",
        "frequency": "Annual cycle",
        "timing": "Weak: Jan-Feb, Oct-Nov. Strong: May-Jul (driving season). Variable: Aug-Sep (hurricanes)",
        "impact_window": "Gradual, over weeks",
        "direction": "High season (May-Jul) → DOWN, Low season (Jan-Feb, Oct-Nov) → UP",
        "magnitude": "MEDIUM",
        "mechanism": "Seasonal refinery demand patterns affect Nigerian Bonny Light and Forcados crude grades.",
    },
    {
        "category": "Oil/Energy",
        "event": "Refinery Turnaround Seasons",
        "frequency": "Twice yearly",
        "timing": "Spring: Mar-May (peak Apr), Fall: Sep-Nov (peak Oct)",
        "impact_window": "2-4 weeks before peak maintenance",
        "direction": "UP — reduced crude demand depresses prices",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Refineries shut for maintenance, reducing crude intake and spot demand for light sweet crudes like Nigeria's.",
    },
    {
        "category": "Oil/Energy",
        "event": "LNG Price Seasonality",
        "frequency": "Annual",
        "timing": "Peak: Nov-Feb (winter heating). Trough: Apr-Sep",
        "impact_window": "Gradual",
        "direction": "High LNG prices → DOWN (Nigeria is major LNG exporter), Low → less FX support",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "NLNG export revenue is Nigeria's second-largest FX earner. Winter price spikes boost dollar earnings.",
    },
    # =====================================================================
    # CHINA
    # =====================================================================
    {
        "category": "China",
        "event": "Chinese New Year / Spring Festival",
        "frequency": "Annual",
        "timing": "Late Jan to mid-Feb (lunar calendar). Factory shutdowns 2 weeks before, restart 2-3 weeks after",
        "impact_window": "2-3 weeks before through 3 weeks after (4-6 week disruption)",
        "direction": "UP — reduced manufacturing demand temporarily depresses oil/commodity prices",
        "magnitude": "MEDIUM",
        "mechanism": "Factory shutdowns reduce crude oil imports and industrial commodity demand. Slows Chinese imports from Nigeria.",
    },
    {
        "category": "China",
        "event": "China Golden Week (National Day)",
        "frequency": "Annual",
        "timing": "October 1-7 (sometimes extended to ~10 days)",
        "impact_window": "2-3 days before through 1 week after",
        "direction": "Mildly UP — reduced trading activity and factory output",
        "magnitude": "SMALL",
        "mechanism": "Factory and market closures temporarily reduce commodity demand.",
    },
    {
        "category": "China",
        "event": "China NBS Manufacturing PMI",
        "frequency": "Monthly",
        "timing": "Last day of month (or 1st of next month)",
        "impact_window": "Same day, sets tone for Asian commodity trading",
        "direction": "Above 50 → DOWN (oil demand up), Below 50 → UP (oil demand down)",
        "magnitude": "MEDIUM",
        "mechanism": "Leading indicator of global manufacturing and commodity demand.",
    },
    {
        "category": "China",
        "event": "China GDP Release",
        "frequency": "Quarterly",
        "timing": "~15th-18th of Jan, Apr, Jul, Oct",
        "impact_window": "Same day",
        "direction": "Strong → DOWN (commodity demand up), Weak → UP",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Signals trajectory of the world's largest commodity importer.",
    },
    {
        "category": "China",
        "event": "China Trade Balance / Crude Oil Imports",
        "frequency": "Monthly",
        "timing": "~7th-10th of month",
        "impact_window": "Same day",
        "direction": "Strong imports (esp. crude) → DOWN, Weak → UP",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "China's crude oil import volumes directly affect global oil demand. China is world's largest crude importer.",
    },
    {
        "category": "China",
        "event": "PBoC Rate Decisions (LPR/MLF)",
        "frequency": "Monthly (LPR: 20th; MLF: mid-month)",
        "timing": "20th of each month",
        "impact_window": "Same day",
        "direction": "Rate cut/stimulus → DOWN (commodity demand expectations rise), Tightening → UP",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "Monetary easing stimulates Chinese economic activity and commodity demand.",
    },
    # =====================================================================
    # GLOBAL CENTRAL BANKS & INSTITUTIONS
    # =====================================================================
    {
        "category": "Global",
        "event": "ECB Rate Decisions",
        "frequency": "8x/year",
        "timing": "Thursdays, 2:15pm CET",
        "impact_window": "Same day, through EUR/USD → DXY",
        "direction": "ECB hawkish → DXY weakens → DOWN, ECB dovish → DXY strengthens → UP",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "EUR is ~57% of DXY. ECB-Fed policy divergence drives EUR/USD which dominates DXY.",
    },
    {
        "category": "Global",
        "event": "Bank of Japan Rate Decisions",
        "frequency": "8x/year",
        "timing": "Varies; typically 2-day meetings ending Friday",
        "impact_window": "Same day; dramatic when BoJ surprises",
        "direction": "BoJ tightening → carry trade unwind → risk-off → UP, Dovish → neutral/supportive",
        "magnitude": "SMALL (usually), LARGE during carry trade unwinds",
        "mechanism": "JPY is a major carry trade funding currency. BoJ tightening forces carry trade unwinding, causing EM sell-offs.",
    },
    {
        "category": "Global",
        "event": "MSCI EM / Frontier Index Rebalancing",
        "frequency": "Quarterly (Feb, May, Aug, Nov)",
        "timing": "Announcement 2-3 weeks before; effective last business day of month",
        "impact_window": "Announcement through effective date",
        "direction": "Weight increase → DOWN (inflows), Decrease → UP. Re-inclusion would be LARGE",
        "magnitude": "SMALL (currently), LARGE if Nigeria re-included",
        "mechanism": "~$1.8T benchmarked to MSCI EM. Index funds mechanically buy/sell based on country weights.",
    },
    {
        "category": "Global",
        "event": "JPMorgan GBI-EM Bond Index Rebalancing",
        "frequency": "Monthly rebalancing, quarterly major reviews",
        "timing": "Last business day of month; reviews announced 6 weeks before",
        "impact_window": "2-4 weeks before effective date for large changes",
        "direction": "Inclusion/weight increase → DOWN (bond inflows), Exclusion → UP",
        "magnitude": "MEDIUM-LARGE (for inclusion/exclusion events)",
        "mechanism": "~$250B+ tracks GBI-EM. Inclusion forces index funds to buy Nigerian bonds, bringing dollar flows.",
    },
    {
        "category": "Global",
        "event": "IMF/World Bank Meetings & Article IV",
        "frequency": "Spring (Apr) and Annual (Oct) meetings; Article IV annual",
        "timing": "Mid-April and mid-October",
        "impact_window": "During meetings and 1-2 weeks after",
        "direction": "Positive review → DOWN, Criticism → UP",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "IMF assessments influence foreign portfolio investor sentiment and rating agency views.",
    },
    {
        "category": "Global",
        "event": "Sovereign Credit Rating Reviews (Moody's/S&P/Fitch)",
        "frequency": "Annual or semi-annual",
        "timing": "Moody's typically Oct-Nov; S&P/Fitch vary",
        "impact_window": "1-2 days before (if expected) through 1-2 weeks after",
        "direction": "Upgrade → DOWN, Downgrade → UP",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Ratings determine index eligibility and which investors can hold Nigerian bonds. Downgrade triggers forced selling.",
    },
    {
        "category": "Global",
        "event": "Carry Trade Unwind Episodes",
        "frequency": "Episodic, seasonal tendency Jul-Aug and Oct",
        "timing": "Most common July-August and October",
        "impact_window": "Sudden; 1-4 weeks",
        "direction": "UP — sharp Naira weakening",
        "magnitude": "MEDIUM-LARGE",
        "mechanism": "Investors borrow in JPY/CHF to invest in high-yield EM. When risk rises, forced liquidation cascades through EM.",
    },
    {
        "category": "Global",
        "event": "VIX Seasonal Spikes",
        "frequency": "Annual pattern",
        "timing": "Historically highest: Sep-Oct. Also Jul-Aug (low liquidity amplifies). Lowest: Nov-Dec, Mar-Apr",
        "impact_window": "Continuous",
        "direction": "VIX > 25-30 → UP (risk-off, EM sell-off), Low VIX → DOWN (risk-on)",
        "magnitude": "MEDIUM",
        "mechanism": "High VIX triggers portfolio de-risking, hitting EM currencies first as investors flee to USD safe haven.",
    },
    {
        "category": "Global",
        "event": "Quarter-End Portfolio Rebalancing (Global)",
        "frequency": "Quarterly",
        "timing": "Last 3-5 trading days of Mar, Jun, Sep, Dec",
        "impact_window": "1 week before quarter-end, most intense last 2-3 days",
        "direction": "If US outperformed EM → DOWN (rebalance sells USD), If EM underperformed → UP",
        "magnitude": "MEDIUM",
        "mechanism": "Pension funds and sovereign wealth funds rebalance fixed allocation targets, creating counter-trend FX flows.",
    },
    # =====================================================================
    # REGIONAL AFRICA
    # =====================================================================
    {
        "category": "Regional",
        "event": "Ghana Cedi (GHS) Crisis / Sharp Moves",
        "frequency": "Episodic; Ghana MPC every 2 months",
        "timing": "Ghana MPC meetings, IMF review dates",
        "impact_window": "Same day for large moves; builds over weeks for trends",
        "direction": "Sharp Cedi depreciation → UP (regional contagion), Stabilization → neutral",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "West African peer. Investor perception of regional FX risk is correlated. Cross-border trade creates smuggling arbitrage.",
    },
    {
        "category": "Regional",
        "event": "South African Rand (ZAR) as EM Proxy",
        "frequency": "Continuous; SARB MPC every 2 months",
        "timing": "SARB: Jan, Mar, May, Jul, Sep, Nov",
        "impact_window": "Same day for large moves",
        "direction": "ZAR weakness (risk-off) → UP, ZAR strength → DOWN",
        "magnitude": "SMALL-MEDIUM",
        "mechanism": "ZAR is the most liquid African currency and serves as an EM/Africa proxy for global fund managers.",
    },
]



# =====================================================================
# SCHEDULING METADATA
# =====================================================================
# Keyed by event name → dict describing when it occurs.
#   "m"      : list of month numbers (1-12), or None = every month
#   "d"      : approximate day-of-month (int), or None
#   "wd"     : list of weekday ints (0=Mon … 6=Sun) — for weekly events
#   "nth_wd" : (weekday, n) — nth occurrence of weekday; n=-1 for last
# If the entry is None the event is irregular / unpredictable.
# =====================================================================

_SCHEDULES: dict[str, dict | None] = {
    # --- Nigeria ---
    "CBN Monetary Policy Committee (MPC)": {"m": [1, 3, 5, 7, 9, 11], "d": 22},
    "CBN FX Interventions / SMIS Auctions": {"m": None, "wd": [0, 1, 2, 3]},
    "Nigerian CPI / Inflation (NBS)": {"m": None, "d": 15},
    "FAAC Monthly Distribution": {"m": None, "d": 22},
    "Company Income Tax (CIT) Deadline": {"m": [3, 6], "d": 30},
    "Petroleum Profit Tax (PPT) Payments": {"m": None, "d": 28},
    "CBN External Reserves Report": {"m": None, "wd": [3, 4]},
    "Nigerian Oil Production Data (NNPC/OPEC)": {"m": None, "d": 15},
    "FGN Bond Auctions (DMO)": {"m": None, "nth_wd": (2, 3)},  # 3rd Wednesday
    "NTB / OMO Auctions": {"m": None, "nth_wd": (2, 1)},  # 1st Wednesday (approx)
    "Eurobond Coupon / Principal Payments": {"m": [2, 7, 11], "d": 15},
    "Government Budget Presentation & Signing": {"m": [10, 11, 12, 1, 2, 3]},
    "Nigerian Elections": None,
    "Fuel Subsidy Removal / PMS Price Adjustments": None,
    "NLNG Dividend Repatriation": {"m": [4, 5, 6], "d": 15},
    "Dangote Refinery Milestones": None,
    # --- Seasonal ---
    "Christmas / New Year Demand Surge": {"m": [10, 11, 12, 1]},
    "Foreign University Tuition Payments (Sep Intake)": {"m": [6, 7, 8, 9]},
    "Foreign University Tuition (Jan Intake)": {"m": [11, 12, 1]},
    "Hajj Season FX Demand": {"m": [4, 5, 6]},
    "Ramadan": {"m": [2, 3]},
    "Eid al-Fitr / Eid al-Adha": {"m": [3, 4, 6, 7]},
    "Diaspora Remittance Peaks": {"m": [3, 4, 8, 9, 11, 12]},
    "Q4 Dollar Demand Surge": {"m": [10, 11, 12]},
    "January Effect": {"m": [1], "d": 7},
    "End-of-Quarter Window Dressing": {"m": [3, 6, 9, 12], "d": 27},
    "Planting Season / Agricultural Imports": {"m": [3, 4, 5, 6]},
    "Cocoa / Agricultural Export Season": {"m": [5, 6, 7, 8, 10, 11, 12, 1]},
    # --- US / Fed ---
    "FOMC Rate Decision": {"m": [1, 3, 5, 6, 7, 9, 11, 12], "d": 18},
    "US CPI (Consumer Price Index)": {"m": None, "d": 12},
    "US Non-Farm Payrolls (NFP)": {"m": None, "nth_wd": (4, 1)},  # 1st Friday
    "Core PCE Price Index (Fed's preferred measure)": {"m": None, "nth_wd": (4, -1)},  # last Friday
    "Jackson Hole Symposium": {"m": [8], "d": 27},
    "Fed Chair Congressional Testimony": {"m": [2, 3, 6, 7], "d": 12},
    "FOMC Minutes": {"m": [1, 2, 4, 5, 7, 8, 10, 11], "d": 8},
    "US GDP (Advance Estimate)": {"m": [1, 4, 7, 10], "d": 28},
    "ISM Services PMI": {"m": None, "d": 3},
    "US Retail Sales": {"m": None, "d": 15},
    "US Treasury Quarterly Refunding (QRA)": {"m": [1, 4, 7, 10], "nth_wd": (2, -1)},
    "US Debt Ceiling Events": None,
    "US Presidential / Midterm Elections": {"m": [11], "d": 5},
    "Year-End Dollar Funding Squeeze": {"m": [11, 12, 1]},
    "US Tax Season": {"m": [3, 4], "d": 10},
    "Employment Cost Index (ECI)": {"m": [1, 4, 7, 10], "d": 28},
    # --- Oil / Energy ---
    "OPEC/OPEC+ Full Ministerial Meetings": {"m": [6, 12], "d": 5},
    "OPEC+ JMMC Monitoring Committee": {"m": [2, 4, 6, 8, 10, 12], "d": 3},
    "US EIA Weekly Petroleum Report": {"m": None, "wd": [2]},  # every Wednesday
    "IEA Monthly Oil Market Report": {"m": None, "d": 13},
    "OPEC Monthly Oil Market Report (MOMR)": {"m": None, "d": 12},
    "Brent Crude Seasonal Pattern": {"m": [1, 2, 5, 6, 7, 8, 9, 10, 11]},
    "Refinery Turnaround Seasons": {"m": [3, 4, 5, 9, 10, 11]},
    "LNG Price Seasonality": {"m": [11, 12, 1, 2]},
    # --- China ---
    "Chinese New Year / Spring Festival": {"m": [1, 2]},
    "China Golden Week (National Day)": {"m": [10], "d": 1},
    "China NBS Manufacturing PMI": {"m": None, "d": 30},
    "China GDP Release": {"m": [1, 4, 7, 10], "d": 16},
    "China Trade Balance / Crude Oil Imports": {"m": None, "d": 8},
    "PBoC Rate Decisions (LPR/MLF)": {"m": None, "d": 20},
    # --- Global ---
    "ECB Rate Decisions": {"m": [1, 3, 4, 6, 7, 9, 10, 12], "d": 15},
    "Bank of Japan Rate Decisions": {"m": [1, 3, 4, 6, 7, 9, 10, 12], "d": 20},
    "MSCI EM / Frontier Index Rebalancing": {"m": [2, 5, 8, 11], "d": 28},
    "JPMorgan GBI-EM Bond Index Rebalancing": {"m": None, "d": 28},
    "IMF/World Bank Meetings & Article IV": {"m": [4, 10], "d": 15},
    "Sovereign Credit Rating Reviews (Moody's/S&P/Fitch)": {"m": [4, 5, 10, 11]},
    "Carry Trade Unwind Episodes": {"m": [7, 8, 10]},
    "VIX Seasonal Spikes": {"m": [7, 8, 9, 10]},
    "Quarter-End Portfolio Rebalancing (Global)": {"m": [3, 6, 9, 12], "d": 28},
    # --- Regional ---
    "Ghana Cedi (GHS) Crisis / Sharp Moves": None,
    "South African Rand (ZAR) as EM Proxy": {"m": [1, 3, 5, 7, 9, 11], "d": 15},
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> int | None:
    """Return the day-of-month for the *n*-th occurrence of *weekday*.

    *n* = 1 → first, 2 → second, …, -1 → last.
    Returns None if the occurrence doesn't exist.
    """
    weeks = _cal.monthcalendar(year, month)
    if n > 0:
        count = 0
        for week in weeks:
            if week[weekday] != 0:
                count += 1
                if count == n:
                    return week[weekday]
    elif n == -1:
        for week in reversed(weeks):
            if week[weekday] != 0:
                return week[weekday]
    return None


def get_month_events(
    year: int, month: int, *, category: str | None = None, magnitude: str | None = None
) -> dict[int, list[dict]]:
    """Return ``{day_number: [events]}`` for the given month.

    Events that span the whole month (seasonal / no specific day) are placed
    on day **0** — the caller can display them in a banner above the grid.
    """
    result: dict[int, list[dict]] = defaultdict(list)
    max_day = _cal.monthrange(year, month)[1]

    for evt in EVENTS:
        if category and category != "All" and evt["category"] != category:
            continue
        if magnitude and magnitude != "All" and magnitude.upper() not in evt["magnitude"].upper():
            continue

        sched = _SCHEDULES.get(evt["event"])
        if sched is None:
            # Irregular — skip from calendar (shown in a separate list)
            continue

        months = sched.get("m")
        if months is not None and month not in months:
            continue

        # --- resolve to day(s) ---
        if "wd" in sched:
            # Weekly recurring on specific weekdays
            for week in _cal.monthcalendar(year, month):
                for wd in sched["wd"]:
                    if week[wd] != 0:
                        result[week[wd]].append(evt)
        elif "nth_wd" in sched:
            wd, n = sched["nth_wd"]
            day = _nth_weekday(year, month, wd, n)
            if day:
                result[day].append(evt)
        elif "d" in sched and sched["d"] is not None:
            day = min(sched["d"], max_day)
            result[day].append(evt)
        else:
            # Seasonal / range — place on day 0 (whole-month banner)
            result[0].append(evt)

    return dict(result)


def get_events_by_category(category: str | None = None) -> list[dict]:
    if category is None or category == "All":
        return EVENTS
    return [e for e in EVENTS if e["category"] == category]


def get_categories() -> list[str]:
    return sorted(set(e["category"] for e in EVENTS))
