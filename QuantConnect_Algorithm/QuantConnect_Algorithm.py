import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.linalg import solve as linsolver

class MarkowitzMinimumVolatilityInstabilityAnalyses(QCAlgorithm):

    def Initialize(self):
        
        # -----------------
        # Input Parameters.
        # -----------------
        
        # Start date of the backtesting.
        self.SetStartDate(2020, 1, 1)
        # End date of the backtesting.
        self.SetEndDate(2020, 12, 31)
        # Initial account balance.
        self.SetCash(1000000)
        # Benchmark for backtesting.
        self.SetBenchmark("SPY")
        
        # Optimisation model for the portfolio construction. Available options:
        #   - "Equal Weight"
        #   - "Inverse Variance"
        #   - "Markowitz Minimum Volatility"
        #   - "Hierarchical Risk Parity"
        self.OptimisationModel = "Markowitz Minimum Volatility"
        # Normalise the allocation vectors computed so that the weights in the portfolio sum up to 1.
        NormaliseAllocationVector = True
        # Number of past days' returns data to consider for the portfolio construction.
        LookbackPeriods = 60
        # New York time of the first day in the month at which the portfolio is rebalanced.
        RebalancingTime = "10:00"
        # Set the first month to start trading. If set to 0, enter the first portfolio on the first day of the backtesting.
        FirstTradingMonth = 0
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
        
        # Set the internal clock of the algorithm to the New York timezone.
        self.SetTimeZone("America/New_York")
        
        # Create a list with QC Symbol objects of the selected assets.
        QCSymbols = [Symbol.Create(asset, SecurityType.Equity, Market.USA) for asset in SelectedAssets]
        # Set the data resolution for all instruments in the algorithm to 1-minute resolution for the portfolio construction model to be triggered at the exact time chosen.
        self.UniverseSettings.Resolution = Resolution.Minute
        # Pass a custom initialiser for the instruments in the algorithm in order to adjust the leverage and the normalisation mode of each one.
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        # Set the universe selection model to manual, and add the symbols of the assets to consider in the portfolio.
        self.AddUniverseSelection(ManualUniverseSelectionModel(QCSymbols))
        # Turn off margin calls. This is needed to be able to simulate the theoretical Markowitz' minimum volatility portfolio as it is.
        self.Portfolio.MarginCallModel = MarginCallModel.Null
        
        # Set the alpha model to generate constant insights of type price, upwards direction and 1 month duration.
        self.SetAlpha(AtMonthStartAlphaModel(RebalancingTime, FirstTradingMonth))
        
        # Set the portfolio construction model to the custom model "PortfolioOptimisationModel" with the chosen optimisation method, lookback periods and normalisation setting.
        self.SetPortfolioConstruction(PortfolioOptimisationModel(self.OptimisationModel, LookbackPeriods, NormaliseAllocationVector))
        
        # Set the risk management model to the null model that does nothing.
        self.SetRiskManagement(NullRiskManagementModel())
        
        # Set the execution model to a custom model that optimises the margin used when rebalancing.
        self.SetExecution(MarginOptimisedExecutionModel())
        
        # Define global variables to save data generated by the portfolio optimisation method for later analyses.
        # The covariance matrices and allocation vectors are saved as if index by symbol alphabetically sorted.
        global CovarianceMatricesExport
        CovarianceMatricesExport = dict()
        global AllocationVectorsExport
        AllocationVectorsExport = dict()
        global TurnoverFiguresExport
        TurnoverFiguresExport = dict()

    def OnData(self, data):
        ''' OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        pass
        
    def CustomSecurityInitializer(self, security):
        '''Custom security initialiser to set the leverage of each security higher to allow portfolios with individual allocations higher than 2 to be simulated.
        Moreover, the data normalisation mode is set to total return to account for ETP distributions.
        Arguments:
            security: security object of the instrument to initialise.
        '''
        security.SetLeverage(3)
        security.SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
        
    def OnOrderEvent(self, orderEvent):
        '''This method is executed every time an order is sent in order to log its details.
        Arguments:
            orderEvent: OrderEvent object of the new order sent.
        '''
        
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if orderEvent.Status == OrderStatus.Filled: 
            self.Log("{0} - {1}: {2}".format(self.Time, order.Type, orderEvent))
        
    def OnEndOfAlgorithm(self):
        '''This method is executed at the very end of the backtesting period in order to liquidate all open positions
        and save the data generated by the optimisation methods in the ObjectStore.
        '''
        
        # Close all open positions
        self.Liquidate()
        
        modelname = ""
        
        if self.OptimisationModel == "Equal Weight":
            modelname = "EWP"
            
        elif self.OptimisationModel == "Inverse Variance":
            modelname = "IVP"
            
        elif self.OptimisationModel == "Markowitz Minimum Volatility":
            modelname = "MMVP"
            
        elif self.OptimisationModel == "Hierarchical Risk Parity":
            modelname = "HRP"
        
        # If covariance matrices were generated, save them.
        if CovarianceMatricesExport:
            covmatrices = pd.DataFrame.from_dict(CovarianceMatricesExport, orient="index")
            self.ObjectStore.Save(modelname+" Covariance Matrices", covmatrices.to_json())
        
        # If allocation vectors were generated, save them.
        if AllocationVectorsExport:
            allovectors = pd.DataFrame.from_dict(AllocationVectorsExport, orient="index")
            self.ObjectStore.Save(modelname+" Allocation Vectors", allovectors.to_json())
            
        # If turnover figures were generated, save them.
        if TurnoverFiguresExport:
            tovector = pd.DataFrame.from_dict(TurnoverFiguresExport, orient="index")
            self.ObjectStore.Save(modelname+" Turnover", tovector.to_json())
        
class AtMonthStartAlphaModel(AlphaModel):
    '''Custom AlphaModel class that emits constant insights of type price, upwards direction and 1 month duration on the first day of each month at the chosen rebalancing time.
    '''
    
    def __init__(self, rebalancingTime, currentMonth=0):
        # List of symbols for which insights are to be emitted.
        self.symbols = []
        # Last month for which insights were emitted.
        self.lastInsightMonth = currentMonth
        # Hour of the day at which to emit insights.
        self.rebalancingHour = int(rebalancingTime.split(":")[0])
        # Minute of the hour at which to emit insights.
        self.rebalancingMinute = int(rebalancingTime.split(":")[1])
    
    def OnSecuritiesChanged(self, algorithm, changes):
        '''This method is executed whenever securities are added or removed from the algorithm's universe. Update "self.symbols" accordingly if this happens.
        Arguments:
            algorithm: algorithm object that calls this method.
            changes: list of changed securities.
        '''
        
        for security in changes.AddedSecurities:
            if security.Symbol not in self.symbols:
                self.symbols.append(security.Symbol)
                
        for security in changes.RemovedSecurities:
            if security.Symbol in self.symbols:
                self.symbols.remove(security.Symbol)
                
    def Update(self, algorithm, data):
        '''This method is executed on every new piece of data that arrives. If it's rebalancing time, emit insights for each of the symbols in "self.symbols".
        Arguments:
            algorithm: algorithm object that calls this method.
            data: new piece of data that arrived.
        '''
        
        if len(self.symbols) > 1:
        
            currentTime = algorithm.Time
            
            if currentTime.month != self.lastInsightMonth:
                # It is the first day of a new month.
                if currentTime.hour == self.rebalancingHour and currentTime.minute == self.rebalancingMinute:
                    # New rebalancing point.
                    algorithm.Log("New rebalancing point.")
                    self.lastInsightMonth = currentTime.month
                    # Emit "up" insights.
                    return Insight.Group([Insight.Price(symbol, timedelta(minutes=1), InsightDirection.Up) for symbol in self.symbols])
        
        return []
        
class PortfolioOptimisationModel(PortfolioConstructionModel):
    '''PortfolioConstructionModel object that computes optimal portfolios according to the equal weight, inverse variance, Markowitz minimum variance or hierarchical risk parity methods.
    '''
    
    def __init__(self, optimisationModel, lookbackPeriods, normalise=True):
        # Optimisation model for the portfolio construction. Implemented options are:
        #   - "Equal Weight"
        #   - "Inverse Variance"
        #   - "Markowitz Minimum Volatility"
        #   - "Hierarchical Risk Parity"
        self.OptimisationModel = optimisationModel
        # Number of past days' returns data to consider for the portfolio construction.
        self.LookbackPeriods = lookbackPeriods
        # Normalise the allocation vectors computed so that the weights in the portfolio sum up to 1.
        self.NormaliseAllocationVector = normalise
        
    def CreateTargets(self, algorithm, insights):
        '''This method is executed every time new insights are emitted by the "AtMonthStartAlphaModel" object.
        If new insights are emitted, compute optimal portfolio for the symbols of the insights according to the model "self.OptimisationModel".
        Arguments:
            algorithm: algorithm object that calls this method.
            insights: list of insights emitted by the "AtMonthStartAlphaModel" object
        '''
        
        if len(insights) > 0:
            algorithm.Log("Computing portfolio allocation...")
            
            # Get the past returns of the symbols for which insights were received.
            returnsData = self.GetHistoricalDailyReturns(algorithm, [insight.Symbol for insight in insights])
            
            # Compute optimal portfolio according to the model self.OptimisationModel
            # and return a set of targets for the MarginOptimisedExecutionModel object to rebalance the portfolio accordingly.
            if self.OptimisationModel == "Equal Weight":
                return self.ComputeEqualWeightPF(algorithm, returnsData)
                
            elif self.OptimisationModel == "Inverse Variance":
                return self.ComputeInverseVariancePF(algorithm, returnsData)
                
            if self.OptimisationModel == "Markowitz Minimum Volatility":
                return self.ComputeMarkowitzMinVolPF(algorithm, returnsData)
                
            if self.OptimisationModel == "Hierarchical Risk Parity":
                return self.ComputeHRPPF(algorithm, returnsData)
            
        return []
        
    def GetHistoricalDailyReturns(self, algorithm, symbols):
        '''Get past data and compute past daily returns of the symbols in the "symbols" parameter.
        Returns a pandas DataFrame with the symbol names as index sorted alphabetically and the returns of each of the "self.LookbackPeriods" days as columns.
        Arguments:
            algorithm: algorithm object that calls this method.
            symbols: list of Symbol objects for which data must be retrieved.
        '''
        
        historicalReturns = dict()
    
        for symbol in symbols:
            symbolReturns = algorithm.History(symbol, self.LookbackPeriods, Resolution.Daily).loc[symbol, ["close", "open"]]
            symbolReturns = symbolReturns.apply(lambda x: x.loc["close"] / x.loc["open"] - 1, axis=1)
            historicalReturns[str(symbol)] = symbolReturns
    
        return pd.DataFrame.from_dict(historicalReturns, orient="index").sort_index()
        
    def ComputeEqualWeightPF(self, algorithm, returnsData):
        '''Compute equal weight portfolio and send target allocation to the "AtMonthStartAlphaModel" object.
        Arguments:
            algorithm: algorithm object that calls this method.
            returnsData: pandas DataFrame with past daily return data generated by the "self.GetHistoricalDailyReturns" method.
                         This parameter is necessary to know which assets the portfolio must contain.
        '''
        
        numAssets = len(returnsData)
        targets = [PortfolioTarget.Percent(algorithm, symbol, 1./numAssets) for symbol in returnsData.index]
        
        # Save allocation vector.
        AllocationVectorsExport[algorithm.Time] = np.full(numAssets, 1./numAssets)
            
        return targets
    
    def ComputeInverseVariancePF(self, algorithm, returnsData):
        '''Compute inverse variance portfolio and send target allocation to the "AtMonthStartAlphaModel" object.
        
        Arguments:
            algorithm: algorithm object that calls this method.
            returnsData: pandas DataFrame with past daily return data generated by the "self.GetHistoricalDailyReturns" method.
        '''
        
        varmatrix = returnsData.var(axis=1)
        weights = 1 / varmatrix
        if self.NormaliseAllocationVector:
            weights = weights / weights.sum()
            
        targets = [PortfolioTarget.Percent(algorithm, symbol, weights[symbol]) for symbol in weights.index]
        
        # Save allocation vector.
        AllocationVectorsExport[algorithm.Time] = weights
        # Save variance matrix as covariance matrix. All assets assumed to be uncorrelated.
        CovarianceMatricesExport[algorithm.Time] = varmatrix.to_numpy()
            
        return targets
    
    def ComputeMarkowitzMinVolPF(self, algorithm, returnsData):
        '''Compute Markowitz minimum variance portfolio and send target allocation to the "AtMonthStartAlphaModel" object.
        
        Arguments:
            algorithm: algorithm object that calls this method.
            returnsData: pandas DataFrame with past daily return data generated by the "self.GetHistoricalDailyReturns" method.
        '''
        
        # "returnsData" contains the returns of each symbol in rows.
        # pandas cov() method compute the covariance of the columns.
        covmatrix = returnsData.T.cov()
        numAssets = covmatrix.shape[0]
        
        # Try to solve the linear system using the Cholesky decomposition of the covariance matrix.
        try:
            weights = linsolver(covmatrix, np.ones(numAssets), assume_a="pos")
            
        # If the Cholesky decomposition fails, the covariance matrix does not have full rank.
        # Solve the linear system with the least-squares method.
        except:
            algorithm.Log("The covariance matrix is singular. Solve linear system with the least-squares method.")
            weights, res, rnk, s = np.linalg.lstsq(covmatrix, np.ones(numAssets))
            
        if self.NormaliseAllocationVector:
            weights = weights / weights.sum()
        
        # Generate Target objects.
        targets = []
        i = 0
        for symbol in returnsData.index:
            targets.append(PortfolioTarget.Percent(algorithm, symbol, weights[i]))
            i = i + 1
            
        # Save allocation vector.
        AllocationVectorsExport[algorithm.Time] = weights
        # Save covariance matrix as vector.
        CovarianceMatricesExport[algorithm.Time] = covmatrix.to_numpy().reshape(numAssets**2)
            
        return targets
    
    def ComputeHRPPF(self, algorithm, returnsData):
        '''Compute hierarchical risk parity portfolio and send target allocation to the "AtMonthStartAlphaModel" object.
        Code from Advances in Financial Machine Learning, Marcos López de Prado.
        Small changes to the code from pages 240-242 applied to fit QuantConnect's API and the general implementation structure of the program.
        
        Arguments:
            algorithm: algorithm object that calls this method.
            returnsData: pandas DataFrame with past daily return data generated by the "self.GetHistoricalDailyReturns" method.
        '''
        
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
        
        if self.NormaliseAllocationVector:
            w = w / w.sum()
            
        targets = [PortfolioTarget.Percent(algorithm, symbol, w[symbol]) for symbol in w.index]
        
        # Save allocation vector.
        AllocationVectorsExport[algorithm.Time] = w
        
        return targets
        
class MarginOptimisedExecutionModel(ExecutionModel):
    '''Custom ExecutionModel class that optimises used margin on rebalancing by first rebalancing positions which will be reduced
    and then rebalancing positions which will increased.
    '''
    
    def Execute(self, algorithm, targets):
        
        # If there are targets, start rebalancing.
        if len(targets) > 0:
            
            algorithm.Log("Rebalancing portfolio...")
            
            # Compute the change in allocation to each symbol according to the new targets.
            orderQuantities = []
            
            for target in targets:
                open_quantity = sum([x.Quantity for x in algorithm.Transactions.GetOpenOrders(target.Symbol)])
                existing = algorithm.Securities[target.Symbol].Holdings.Quantity + open_quantity
                if target.Quantity - existing != 0:
                    orderQuantities.append((target.Symbol, target.Quantity - existing))
                
            # Sort the changes in allocation.
            sortedOrderQuantities = sorted(orderQuantities, key=lambda target: target[1])
            
            # Save the absolute change in allocation to compute the period's turnover.
            periodTurnover = 0
            
            # Rebalance positions by first tackling the ones that are reduced and the the ones that need to be increased.
            for (symbol, quantity) in sortedOrderQuantities:
                algorithm.MarketOrder(symbol, quantity)
                # Save the absolute change in allocation.
                periodTurnover = periodTurnover + np.abs(quantity)
                
            # Compute and save the relative change in allocation (turnover).
            TurnoverFiguresExport[algorithm.Time] = periodTurnover / algorithm.Portfolio.TotalPortfolioValue
            
        pass