import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

# TODO: add logs and messages.
# TODO: add comments.

class MarkowitzMinimumVolatilityInstabilityAnalyses(QCAlgorithm):

    def Initialize(self):
        
        # -----------------
        # Input Parameters.
        # -----------------
        # Start date of the backtesting. Use a date one month before the first month in which a portfolio is constructed.
        self.SetStartDate(2020, 1, 1)
        # End date of the backtesting.
        self.SetEndDate(2020, 12, 31)
        # Initial account balance.
        self.SetCash(1000000)
        
        # Optimisation model for the portfolio construction.
        # Options: "Equal Weight", "Inverse Variance", "Markowitz Minimum Volatility", "Hierarchical Risk Parity"
        self.optimisationModel = "Hierarchical Risk Parity"
        # Constrain portfolio construction to long-only portfolios.
        # Note: if the optimisation model is Markowitz Minimum Volatility, portfolios are NEVER long-only constrained.
        self.longOnlyBias = True
        # Normalise the allocation vectors computed so that the weights in the portfolio sum up to 1.
        NormaliseAllocationVector = True
        # Number of past days' data to consider for the portfolio construction. If set to 0, no override.
        LookbackPeriodsOverride = 60
        # New York time of the first day in the month at which the portfolio is rebalanced.
        RebalancingTime = "10:00"
        # List with the assets to consider in the portfolio.
        SelectedAssets = [
                    # Equity ETFs
                    "EXI",      # iShares Global Industrials ETF
                    "FILL",     # iShares MSCI Global Energy Producers ETF
                    "IGF",      # iShares Global Infrastructure ETF
                    "IXG",      # iShares Global Financials ETF
                    "IXJ",      # iShares Global Healthcare ETF
                    "IXN",      # iShares Global Tech ETF
                    "IXP",      # iShares Global Comm Services ETF
                    "JXI",      # iShares Global Utilities ETF
                    "KXI",      # iShares Global Consumer Staples ETF
                    "MXI",      # iShares Global Materials ETF
                    "RXI",      # iShares Global Consumer Discretionary ETF
                    # Fixed Income ETFs
                    "IAGG",     # iShares Core International Aggregate Bond ETF
                    "ILTB",     # iShares Core 10+ Year USD Bond ETF
                    "IMTB",     # iShares Core 5-10 Year USD Bond ETF
                    "ISTB",     # iShares Core 1-5 Year USD Bond ETF
                    # Real Estate
                    "REET",     # iShares Global REIT ETF
                    # Commodities
                    "DBA",      # Invesco DB Agriculture Fund
                    "DBB",      # Invesco DB Base Metals Fund
                    "DBE",      # Invesco DB Energy Fund
                    "DBP"       # Invesco DB Precious Metals Fund
                    ]
        # ------------------------
        # End of Input Parameters.
        # ------------------------
        
        self.SetTimeZone("America/New_York")
        
        # Create a list with QC Symbol objects of the selected assets.
        QCSymbols = [Symbol.Create(asset, SecurityType.Equity, Market.USA) for asset in SelectedAssets]
        # Set the data resolution for all instruments in the algorithm to minute resolution for the portfolio construction model to be triggered at the exact time chosen.
        self.UniverseSettings.Resolution = Resolution.Minute
        # Pass a custom initialiser for the instruments in the algorithm in order to adjust the leverage and the normalisation mode of each one.
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        # Set the universe selection model to manual, and add the symbols of the assets to consider in the portfolio.
        self.AddUniverseSelection(ManualUniverseSelectionModel(QCSymbols))
        # Turn off margin calls. This is needed to be able to simulate the theoretical Markowitz' minimum volatility portfolio as it is.
        self.Portfolio.MarginCallModel = MarginCallModel.Null
        
        # Set the alpha model to generate constant insights of type price, upwards direction and 1 month duration.
        self.SetAlpha(AtMonthStartAlphaModel(RebalancingTime, 0))
        
        LookbackPeriods = int((2/3) * len(SelectedAssets)*(len(SelectedAssets) + 1))
        if LookbackPeriodsOverride != 0:
            LookbackPeriods = LookbackPeriodsOverride
        # Set the portfolio construction model to the custom model PortfolioOptimisationModel with the chosen optimisation method, lookback periods direction constrain and normalisation setting.
        self.SetPortfolioConstruction(PortfolioOptimisationModel(self.optimisationModel, LookbackPeriods, self.longOnlyBias, NormaliseAllocationVector))
        
        # Set the risk management model to the null model that does nothing.
        self.SetRiskManagement(NullRiskManagementModel())
        
        # Set the execution model to a custom model that optimises the margin used when rebalancing.
        self.SetExecution(MarginOptimisedExecutionModel())
        
        global CovarianceMatricesExport
        CovarianceMatricesExport = dict()
        global AllocationVectorsExport
        AllocationVectorsExport = dict()

    def OnData(self, data):
        ''' OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        pass
        
    def CustomSecurityInitializer(self, security):
        '''
        Custom security initialiser to set the leverage of each security higher to allow portfolios with individual allocations higher than 2 to be simulated.
        Moreover, the data normalisation mode is set to total return to account for ETP distributions.
        Arguments:
            security: security object of the instrument to initialise.
        '''
        # TODO: update leverage
        security.SetLeverage(3)
        security.SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
        
    def OnOrderEvent(self, orderEvent):
        #https://www.quantconnect.com/docs/algorithm-reference/trading-and-orders#Trading-and-Orders-Tracking-Order-Events
        #https://www.quantconnect.com/docs/algorithm-reference/reality-modelling#Reality-Modelling-Fill-Models
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if orderEvent.Status == OrderStatus.Filled: 
            self.Log("{0}: {1}: {2}".format(self.Time, order.Type, orderEvent)) #orderEvent.OrderFee
        
    def OnEndOfAlgorithm(self):
        
        # Close all open positions
        self.Liquidate()
        
        modelname = ""
                
        if self.optimisationModel == "Markowitz Minimum Volatility":
            modelname = "MMVP"
        
        else:
            if self.optimisationModel == "Equal Weight":
                modelname = "EWP"
                
            if self.optimisationModel == "Inverse Variance":
                modelname = "IVP"
        
            if self.optimisationModel == "Hierarchical Risk Parity":
                modelname = "HRP"
                
            if self.longOnlyBias:
                modelname = modelname + " long-only"
                
            else:
                modelname = modelname + " long-short"
        
        if CovarianceMatricesExport:
            covmatrices = pd.DataFrame.from_dict(CovarianceMatricesExport, orient="index")
            self.ObjectStore.Save(modelname+" Covariance Matrices", covmatrices.to_json())
        
        if AllocationVectorsExport:
            allovectors = pd.DataFrame.from_dict(AllocationVectorsExport, orient="index")
            self.ObjectStore.Save(modelname+" Allocation Vectors", allovectors.to_json())
        
class AtMonthStartAlphaModel(AlphaModel):
    '''
    Custom AlphaModel class that emits constant insights of type price, upwards direction and 1 month duration on the first day of each month at the chosen rebalancing time.
    
    '''
    def __init__(self, rebalancingTime, currentMonth=0):
        self.symbols = []
        self.lastInsightMonth = currentMonth
        self.rebalancingHour = int(rebalancingTime.split(":")[0])
        self.rebalancingMinute = int(rebalancingTime.split(":")[1])
    
    def OnSecuritiesChanged(self, algorithm, changes):
        
        for security in changes.AddedSecurities:
            if security.Symbol not in self.symbols:
                self.symbols.append(security.Symbol)
                
        for security in changes.RemovedSecurities:
            if security.Symbol in self.symbols:
                self.symbols.remove(security.Symbol)
                
    def Update(self, algorithm, data):
        
        if len(self.symbols) > 1:
        
            currentTime = algorithm.Time
            
            if currentTime.month != self.lastInsightMonth:
                # It is the first day of a new month.
                if currentTime.hour == self.rebalancingHour and currentTime.minute == self.rebalancingMinute:
                    # New rebalancing point.
                    algorithm.Debug("New rebalancing point.")
                    self.lastInsightMonth = currentTime.month
                    # Emit "up" insights.
                    return Insight.Group([Insight.Price(symbol, timedelta(minutes=1), InsightDirection.Up) for symbol in self.symbols])
        
        return []
        
class PortfolioOptimisationModel(PortfolioConstructionModel):
    
    def __init__(self, optimisationModel, lookbackPeriods, longonly=False, normalise=True):
        self.optimisationModel = optimisationModel
        self.lookbackPeriods = lookbackPeriods
        self.longonly = longonly
        self.normaliseAllocationVector = normalise
        
    def CreateTargets(self, algorithm, insights):
        if len(insights) > 0:
            algorithm.Debug("Computing portfolio allocation...")
            
            returnsData = self.GetHistoricalDailyReturns(algorithm, [insight.Symbol for insight in insights])
            
            if self.optimisationModel == "Equal Weight":
                return self.ComputeEqualWeightPF(algorithm, returnsData)
            
            if self.optimisationModel == "Inverse Variance":
                return self.ComputeInverseVariancePF(algorithm, returnsData)
                
            if self.optimisationModel == "Markowitz Minimum Volatility":
                return self.ComputeMarkowitzMinVolPF(algorithm, returnsData)
                
            if self.optimisationModel == "Hierarchical Risk Parity":
                return self.ComputeHRPPF(algorithm, returnsData)
            
        return []
        
    def GetHistoricalMonthlyReturns(self, algorithm, symbols):
        historicalReturns = dict()
        total_return = lambda x: x.loc[x.index.max(), "close"] / x.loc[x.index.min(), "open"] - 1
    
        for symbol in symbols:
            symbolReturns = algorithm.History(symbol, timedelta(days=31 * self.lookbackPeriods), Resolution.Daily).loc[symbol, ["close", "open"]]
            symbolReturns = symbolReturns.groupby([symbolReturns.index.year, symbolReturns.index.month]).apply(total_return)
            historicalReturns[str(symbol)] = symbolReturns.tail(self.lookbackPeriods)
    
        return pd.DataFrame.from_dict(historicalReturns, orient="index").sort_index()
        
    def GetHistoricalDailyReturns(self, algorithm, symbols):
        historicalReturns = dict()
    
        for symbol in symbols:
            symbolReturns = algorithm.History(symbol, self.lookbackPeriods, Resolution.Daily).loc[symbol, ["close", "open"]]
            symbolReturns = symbolReturns.apply(lambda x: x.loc["close"] / x.loc["open"] - 1, axis=1)
            historicalReturns[str(symbol)] = symbolReturns
    
        return pd.DataFrame.from_dict(historicalReturns, orient="index").sort_index()
        
    def ComputeEqualWeightPF(self, algorithm, returnsData):
        
        targets = []
        
        weights = returnsData.mean(axis=1).apply(np.sign)
        
        if self.longonly:
            weights = weights.apply(lambda x: 1)
        
        weights = weights / weights.sum()
        targets = [PortfolioTarget.Percent(algorithm, symbol, weights[symbol]) for symbol in weights.index]
            
        # TODO: update to save pandas, not list of targets
        AllocationVectorsExport[algorithm.Time] = weights
            
        return targets
    
    def ComputeInverseVariancePF(self, algorithm, returnsData):
        
        weights = 1 / returnsData.var(axis=1)
        
        if not self.longonly:
            bias = returnsData.mean(axis=1).apply(np.sign)
            weights = np.multiply(weights, bias)
            
        if self.normaliseAllocationVector:
            weights = weights / weights.sum()
            
        targets = [PortfolioTarget.Percent(algorithm, symbol, weights[symbol]) for symbol in weights.index]
        
        AllocationVectorsExport[algorithm.Time] = weights
            
        return targets
    
    def ComputeMarkowitzMinVolPF(self, algorithm, returnsData):
        covmatrix = returnsData.T.cov()
        
        # TODO: save condition numbers.
        #algorithm.Debug(np.linalg.cond(covmatrix))
        numAssets = covmatrix.shape[0]
        
        #covInv = np.linalg.inv(covmatrix)
        #weights = np.dot(covInv, np.ones(numAssets))
        weights = np.linalg.solve(covmatrix, np.ones(numAssets))
        weights = weights / weights.sum()
        
        targets = []
        i = 0
        for symbol in returnsData.index:
            targets.append(PortfolioTarget.Percent(algorithm, symbol, weights[i]))
            i = i + 1
            
        # TODO: save weights with columns names of the symbols.
        AllocationVectorsExport[algorithm.Time] = weights
        CovarianceMatricesExport[algorithm.Time] = covmatrix.to_numpy().reshape(numAssets**2)
            
        return targets
    
    def ComputeHRPPF(self, algorithm, returnsData):
        # Code from Advances in Financial Machine Learning, Marcos López de Prado.
        # Small changes to the code from pages 240-242 applied to fit QuantConnect's API and the general implementation structure of the program.
    
        # pandas cov() and corr() method compute the covariance and correlation of the columns.
        cov, corr = returnsData.T.cov(), returnsData.T.corr()
    
        # 3) cluster
        # Compute distance matrix based on correlation, where 0 <= d[i,j] <= 1.
        dist = ((1 - corr) / 2.)**.5
        link = sch.linkage(dist, "single")
        # Sort clustered items by distance.
        link = link.astype(int)
        sortIx = pd.Series([link[-1,0], link[-1,1]])
        numItems= link[-1, 3] # Number of original items.
        
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0]*2, 2) # Make space.
            df0 = sortIx[sortIx >= numItems] # Find clusters.
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0] # Item 1.
            df0 = pd.Series(link[j, 1], index=i+1)
            sortIx = sortIx.append(df0) # Item 2.
            sortIx = sortIx.sort_index() # Re-sort.
            sortIx.index = range(sortIx.shape[0]) # Re-index
        sortIx = sortIx.tolist()
        sortIx = corr.index[sortIx].tolist() # Recover labels.
        #df0 = corr.loc[sortIx, sortIx] # Reorder.
    
        #4) Capital allocation
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx] # Initialise all items in one cluster
        while len(cItems) > 0:
            cItems = [i[int(j):int(k)] for i in cItems for j, k in ((0, len(i) / 2),
                                                          (len(i) / 2, len(i))) if len(i) > 1] # Bi-section.
                
            for i in range(0, len(cItems), 2): # Parse in pairs.
                cItems0 = cItems[i] # Cluster 1
                cItems1 = cItems[i+1] # Cluster 2
                # Compute variance per cluster.
                cov0_ = cov.loc[cItems0, cItems0] # Matrix slice
                w0_ = 1. / np.diag(cov0_)
                w0_ = w0_ / w0_.sum()
                w0_ = w0_.reshape(-1, 1)
                cVar0 = np.dot(np.dot(w0_.T, cov0_), w0_)[0, 0]
    
                cov1_ = cov.loc[cItems1, cItems1] # Matrix slice
                w1_ = 1. / np.diag(cov1_)
                w1_ = w1_ / w1_.sum()
                w1_ = w1_.reshape(-1, 1)
                cVar1 = np.dot(np.dot(w1_.T, cov1_), w1_)[0, 0]
    
                alpha = 1 - cVar0 / (cVar0+cVar1)
                w[cItems0] *= alpha # Weight 1
                w[cItems1] *= 1 - alpha
                
        if not self.longonly:
            bias = returnsData.mean(axis=1).apply(np.sign)
            w = np.multiply(w, bias)
        
        if self.normaliseAllocationVector:
            w = w / w.sum()
            
        targets = [PortfolioTarget.Percent(algorithm, symbol, w[symbol]) for symbol in w.index]
        
        AllocationVectorsExport[algorithm.Time] = w
        
        return targets
        
class MarginOptimisedExecutionModel(ExecutionModel):
    
    def Execute(self, algorithm, targets):
        
        algorithm.Debug("Rebalancing portfolio...")
        # TODO: track turnover
        
        orderQuantities = []
        
        for target in targets:
            open_quantity = sum([x.Quantity for x in algorithm.Transactions.GetOpenOrders(target.Symbol)])
            existing = algorithm.Securities[target.Symbol].Holdings.Quantity + open_quantity
            if target.Quantity - existing != 0:
                orderQuantities.append((target.Symbol, target.Quantity - existing))
            
        sortedOrderQuantities = sorted(orderQuantities, key=lambda target: target[1])
        
        for (symbol, quantity) in sortedOrderQuantities:
            algorithm.MarketOrder(symbol, quantity)
        
        pass