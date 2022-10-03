import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.linalg import solve as linsolver
from sklearn.cluster import KMeans

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
        self.OptimisationModel = "Hierarchical Risk Parity"
        
        # Covariance shrinkage method used in the optimisation model.
        # This parameter only affects the "Markowitz Minimum Volatility" optimisation model.
        # Available options:
        #   - None              ->  Do not shrink covariance matrix.
        #   - "Ledoit-Wolf"     ->  Use Ledoit and Wolf constant correlation model.
        #   - "De Nard"         ->  Use De Nard's generalised constant-variance-covariance model.
        self.CovarianceShrinkageMethod = None
        
        # Number of groups in which assets are clustered
        # This parameter only affects "De Nard" covariance shrinkage method.
        self.AssetGroupCount = 4
        
        # Normalise the allocation vectors so that the portfolio weights sum up to 1.
        NormaliseAllocationVector = True
        
        # Number of past days' returns data to consider for the portfolio construction.
        LookbackPeriods = 60
        
        # New York time of the first day in the month at which the portfolio is rebalanced.
        RebalancingTime = "10:00"
        
        # Set the first month to start trading.
        # If set to 0, enter the first portfolio on the first day of the backtesting.
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
        # Set the data resolution for all instruments in the algorithm to 1-minute resolution.
        # The portfolio construction model will so be triggered at the exact time chosen.
        self.UniverseSettings.Resolution = Resolution.Minute
        # Pass a custom initialiser for the instruments in the algorithm.
        # This allows to adjust the leverage and the normalisation mode of each one.
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        # Set the universe selection model to manual, and add the symbols of the selected assets.
        self.AddUniverseSelection(ManualUniverseSelectionModel(QCSymbols))
        # Turn off margin calls.
        # This is needed to be able to simulate the theoretical Markowitz' minimum volatility portfolios.
        self.Portfolio.MarginCallModel = MarginCallModel.Null
        
        # Set the alpha model to generate constant insights the first day of each month.
        self.SetAlpha(AtMonthStartAlphaModel(RebalancingTime, FirstTradingMonth))
        
        # Set the portfolio construction model to the chosen optimisation method.
        if self.OptimisationModel == "Equal Weight":
            self.SetPortfolioConstruction(EqualWeightPortfolio())
            
        elif self.OptimisationModel == "Inverse Variance":
            self.SetPortfolioConstruction(InverseVariancePortfolio(LookbackPeriods,
                                                                   NormaliseAllocationVector
                                                                   ))
            
        elif self.OptimisationModel == "Markowitz Minimum Volatility":
            self.SetPortfolioConstruction(MarkowitzMinVolPortfolio(self.CovarianceShrinkageMethod,
                                                                   self.AssetGroupCount,
                                                                   LookbackPeriods,
                                                                   NormaliseAllocationVector
                                                                   ))
            
        elif self.OptimisationModel == "Hierarchical Risk Parity":
            self.SetPortfolioConstruction(HierarchicalRiskParityPortfolio(LookbackPeriods))
        
        # Set the risk management model to the null model that does nothing.
        self.SetRiskManagement(NullRiskManagementModel())
        
        # Set the execution model to a custom model that optimises the margin used when rebalancing.
        self.SetExecution(MarginOptimisedExecutionModel())
        
        # Define global variables to save data generated by the portfolio optimisation method.
        # All but the turnover figures and asset clusterings are saved indexed by symbol
        # alphabetically sorted.
        global CovarianceMatricesExport
        CovarianceMatricesExport = dict()
        global AllocationVectorsExport
        AllocationVectorsExport = dict()
        global TurnoverFiguresExport
        TurnoverFiguresExport = dict()
        global ShrinkageConstantsExport
        ShrinkageConstantsExport = dict()
        global ShrinkageTargetsExport
        ShrinkageTargetsExport = dict()
        global AssetClusteringsExport
        AssetClusteringsExport = dict()

    def OnData(self, data):
        ''' OnData event is the primary entry point for your algorithm.
        Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        pass
        
    def CustomSecurityInitializer(self, security):
        '''Custom security initialiser to set the leverage of each security higher
        to allow portfolios with individual allocations higher than 2 to be simulated.
        Moreover, the data normalisation mode is set to total return to account for
        ETP distributions.
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
        '''This method is executed at the very end of the backtesting period
        in order to liquidate all open positions and save the data generated
        by the optimisation methods in the ObjectStore.
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
            
            if self.CovarianceShrinkageMethod == "Ledoit-Wolf":
                modelname = modelname + "-LW"
                
            if self.CovarianceShrinkageMethod == "De Nard":
                modelname = modelname + "-DN-K" + str(self.AssetGroupCount)
            
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
            
        # If shrinkage constants were generated, save them.
        if ShrinkageConstantsExport:
            scvector = pd.DataFrame.from_dict(ShrinkageConstantsExport, orient="index")
            self.ObjectStore.Save(modelname+" Shrinkage Constants", scvector.to_json())
            
        # If shrinkage targets were generated, save them.
        if ShrinkageTargetsExport:
            stvectors = pd.DataFrame.from_dict(ShrinkageTargetsExport, orient="index")
            self.ObjectStore.Save(modelname+" Shrinkage Targets", stvectors.to_json())
            
        # If clusterings were generated, save them.
        if AssetClusteringsExport:
            clusterings = pd.DataFrame.from_dict(AssetClusteringsExport, orient="index")
            self.ObjectStore.Save(modelname+" Clusterings", clusterings.to_json())

# -------------------------------
# Alpha insights generator model.
# -------------------------------
      
class AtMonthStartAlphaModel(AlphaModel):
    '''Custom AlphaModel class that emits constant insights of type price, upwards direction
    and 1 month duration on the first day of each month at the chosen rebalancing time.
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
        '''This method is executed whenever securities are added or removed from the
        algorithm's universe. Update "self.symbols" accordingly if this happens.
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
        '''This method is executed on every new piece of data that arrives. If it's
        rebalancing time, emit insights for each of the symbols in "self.symbols".
        Arguments:
            algorithm: algorithm object that calls this method.
            data: new piece of data that arrived.
        '''
        
        if len(self.symbols) > 1:
        
            currentTime = algorithm.Time
            
            if currentTime.month != self.lastInsightMonth:
                # It is the first day of a new month.
                if (currentTime.hour == self.rebalancingHour
                    and currentTime.minute == self.rebalancingMinute):
                       
                    # New rebalancing point.
                    algorithm.Log("New rebalancing point.")
                    self.lastInsightMonth = currentTime.month
                    # Emit "up" insights.
                    return Insight.Group([Insight.Price(symbol,
                                                        timedelta(minutes=1),
                                                        InsightDirection.Up)
                                          for symbol in self.symbols])
        
        return []
        
# ------------------------------
# Portfolio construction models.
# ------------------------------
        
class EqualWeightPortfolio(PortfolioConstructionModel):
    '''PortfolioConstructionModel object that computes the equal weight portfolio.
    '''
        
    def CreateTargets(self, algorithm, insights):
        '''This method is executed every time new insights are emitted by the
        "AtMonthStartAlphaModel" object. If new insights are emitted, compute
        equal weight portfolio for the symbols of the insights.
        Arguments:
            algorithm: algorithm object that calls this method.
            insights: list of insights emitted by the "AtMonthStartAlphaModel" object.
        '''
        
        if len(insights) > 0:
            algorithm.Log("Computing equal weight portfolio allocation...")
            
            # Compute equal weight portfolio.
            numAssets = len(insights)
            targets = [PortfolioTarget.Percent(algorithm, insight.Symbol, 1./numAssets)
                        for insight in insights]
            
            # Save allocation vector.
            AllocationVectorsExport[algorithm.Time] = np.full(numAssets, 1./numAssets)
               
            # Return set of targets for the MarginOptimisedExecutionModel object.
            return targets
            
        return []
        
class InverseVariancePortfolio(PortfolioConstructionModel):
    '''PortfolioConstructionModel object that computes the inverse variance portfolio.
    '''
    
    def __init__(self, lookbackPeriods, normalise=True):
        # Number of past days' returns data to consider for the portfolio construction.
        self.LookbackPeriods = lookbackPeriods
        # Normalise the allocation vectors so that the portfolio weights sum up to 1.
        self.NormaliseAllocationVector = normalise
        
    def CreateTargets(self, algorithm, insights):
        '''This method is executed every time new insights are emitted by the
        "AtMonthStartAlphaModel" object. If new insights are emitted, compute
        inverse variance portfolio for the symbols of the insights.
        Arguments:
            algorithm: algorithm object that calls this method.
            insights: list of insights emitted by the "AtMonthStartAlphaModel" object.
        '''
        
        if len(insights) > 0:
            algorithm.Log("Computing inverse variance portfolio allocation...")
            
            # Get the past returns of the symbols for which insights were received.
            returnsData = dict()
        
            for symbol in [insight.Symbol for insight in insights]:
                symbolReturns = algorithm.History(symbol,
                                                  self.LookbackPeriods,
                                                  Resolution.Daily).loc[symbol, ["close", "open"]]
                symbolReturns = symbolReturns.apply(lambda x: x.loc["close"] / x.loc["open"] - 1,
                                                    axis=1)
                returnsData[str(symbol)] = symbolReturns
            # Create pandas DataFrame with the symbol names as index sorted alphabetically
            # and the returns of each of the "self.LookbackPeriods" days as columns.
            returnsData = pd.DataFrame.from_dict(returnsData, orient="index").sort_index()
            
            # Compute optimal portfolio according to the model self.OptimisationModel
            varmatrix = returnsData.var(axis=1)
            weights = 1 / varmatrix
            if self.NormaliseAllocationVector:
                weights = weights / weights.sum()
                
            targets = [PortfolioTarget.Percent(algorithm, symbol, weights[symbol])
                        for symbol in weights.index]
            
            # Save allocation vector.
            AllocationVectorsExport[algorithm.Time] = weights
            # Save variance matrix as covariance matrix.
            # All assets assumed to be uncorrelated.
            CovarianceMatricesExport[algorithm.Time] = varmatrix.to_numpy()
            
            # Return set of targets for the MarginOptimisedExecutionModel object.
            return targets
            
        return []
        
class MarkowitzMinVolPortfolio(PortfolioConstructionModel):
    '''PortfolioConstructionModel object that computes Markowitz Minimum Volatility portfolio.
    '''
    
    def __init__(self, covmatrixshrinkagemode, assetgroupcount, lookbackPeriods, normalise=True):
        # Covariance shrinkage method used in the optimisation model.
        # This parameter only affects the "Markowitz Minimum Volatility" optimisation model.
        # Available options:
        #   - None              ->  Do not shrink covariance matrix.
        #   - "Ledoit-Wolf"     ->  Use Ledoit and Wolf constant correlation model.
        #   - "De Nard"         ->  Use De Nard's generalised constant-variance-covariance model.
        self.CovarianceShrinkageMethod = covmatrixshrinkagemode
        # Number of groups in which assets are clustered
        # This parameter only affects "De Nard" covariance shrinkage method.
        self.AssetGroupCount = assetgroupcount
        # Number of past days' returns data to consider for the portfolio construction.
        self.LookbackPeriods = lookbackPeriods
        # Normalise the allocation vectors so that the portfolio weights sum up to 1.
        self.NormaliseAllocationVector = normalise
        
    def CreateTargets(self, algorithm, insights):
        '''This method is executed every time new insights are emitted by the
        "AtMonthStartAlphaModel" object. If new insights are emitted, compute
        Markowitz Minimum Volatility portfolio for the symbols of the insights.
        Arguments:
            algorithm: algorithm object that calls this method.
            insights: list of insights emitted by the "AtMonthStartAlphaModel" object.
        '''
        
        if len(insights) > 0:
            algorithm.Log("Computing Markowitz minimum volatility portfolio allocation...")
            
            # Get the past returns of the symbols for which insights were received.
            returnsData = dict()
        
            for symbol in [insight.Symbol for insight in insights]:
                symbolReturns = algorithm.History(symbol,
                                                  self.LookbackPeriods,
                                                  Resolution.Daily).loc[symbol, ["close", "open"]]
                symbolReturns = symbolReturns.apply(lambda x: x.loc["close"] / x.loc["open"] - 1,
                                                    axis=1)
                returnsData[str(symbol)] = symbolReturns
            # Create pandas DataFrame with the symbol names as index sorted alphabetically
            # and the returns of each of the "self.LookbackPeriods" days as columns.
            returnsData = pd.DataFrame.from_dict(returnsData, orient="index").sort_index()
            
            # Compute Markowitz minimum variance portfolio.
            
            # Compute (shrunk) covariance matrix.
            if self.CovarianceShrinkageMethod == "Ledoit-Wolf":
                shrinkageconstant, shrinkagetarget, covmatrix = self.LedoitWolfCovShrink(algorithm, returnsData)
                
            elif self.CovarianceShrinkageMethod == "De Nard":
                shrinkageconstant, shrinkagetarget, covmatrix = self.DeNardCovShrink(algorithm, returnsData)
                
            else:
                # We compute the sample covariance matrix after Bessel's correction factor.
                covmatrix = np.cov(returnsData, rowvar=True)
            
            numAssets = covmatrix.shape[0]
            
            # Try to solve the linear system using the Cholesky decomposition of the covariance matrix.
            try:
                weights = linsolver(covmatrix, np.ones(numAssets), assume_a="pos")
                
            # If the Cholesky decomposition fails, the covariance matrix does not have full rank.
            # Solve the linear system with the least-squares method.
            except:
                algorithm.Log("The covariance matrix is singular.")
                algorithm.Log("Solving linear system with the least-squares method...")
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
            CovarianceMatricesExport[algorithm.Time] = covmatrix.reshape(numAssets**2)
            # Save shrinkage constant and shrinkage target as vectors.
            if self.CovarianceShrinkageMethod != None:
                ShrinkageConstantsExport[algorithm.Time] = shrinkageconstant
                ShrinkageTargetsExport[algorithm.Time] = shrinkagetarget.reshape(numAssets**2)
            
            # Return set of targets for the MarginOptimisedExecutionModel object.
            return targets
            
        return []
        
    def LedoitWolfCovShrink(self, algorithm, returnsData):
        '''This method is computes the constant correlation matrix and the optimal
        shrinkage constant as Ledoit and Wolf describe in their paper
        Honey, I Shrunk the Sample Covariance Matrix. It returns, the shrinkage
        constant, the constant correlation matrix and the srhunk sample covariance
        matrix.
        Arguments:
            algorithm: algorithm object that calls this method.
            returnsData: pandas DataFrame with the past period observations of each
                         asset as rows.
        '''
        
        returnsData = returnsData.to_numpy()
        
        # "returnsData" contains the returns of each symbol in rows.
        numAssets, T = returnsData.shape
        
        # Compute centered returns.
        x = (returnsData.T - returnsData.mean(axis=1)).T
        
        # Compute sample covariance, variance and standard deviation matrices.
        samplecovmatrix = np.cov(returnsData, rowvar=True)
        varmatrix = np.diag(np.diag(samplecovmatrix))
        stdmatrix = np.sqrt(varmatrix)
        stdmatrixinv = np.diag(1/np.diag(stdmatrix))
        corrmatrix = np.dot(stdmatrixinv, np.dot(samplecovmatrix, stdmatrixinv))
        
        # Compute the constant correlation matrix corresponding to the sample cov. matrix.
        rbar = (corrmatrix.sum() - numAssets) / ((numAssets-1) * numAssets)
        constcorrmatrix = np.full((numAssets, numAssets), rbar)
        constcorrmatrix = np.dot(stdmatrix, np.dot(constcorrmatrix, stdmatrix))
        np.fill_diagonal(constcorrmatrix, np.diag(varmatrix))
        
        # Compute pi hat.
        y = np.power(x, 2)
        piMat = np.dot(y, y.T) / T - np.power(samplecovmatrix, 2)
        pihat = piMat.sum()
        
        # Compute rho hat.
        term1 = np.dot(np.power(returnsData, 3), returnsData.T) / T
        term2 = np.dot(varmatrix, samplecovmatrix)
        term3 = np.dot(samplecovmatrix, varmatrix)
        term4 = np.dot(varmatrix, samplecovmatrix)
        thetaMat = term1 - term2 - term3 + term4
        np.fill_diagonal(thetaMat, 0)
        rhohat = np.diag(piMat).sum() + rbar * np.multiply(np.outer(np.diag(stdmatrixinv),
                                                                    np.diag(stdmatrix)),
                                                           thetaMat).sum()
        
        # Compute gamma hat.
        gammahat = np.linalg.norm(constcorrmatrix - samplecovmatrix, "fro") ** 2
        
        # Compute shrinkage constant.
        delta = 1/T * (pihat - rhohat) / gammahat
        
        if delta <= 0:
            algorithm.Log("Shrinkage constant is negative or zero.")
            return 0, constcorrmatrix, samplecovmatrix
        
        elif delta >= 1:
            algorithm.Log("Shrinkage constant is greater than 1.")
            return 1, constcorrmatrix, constcorrmatrix
        
        return delta, constcorrmatrix, (delta * constcorrmatrix + (1-delta) * samplecovmatrix)
        
    def DeNardCovShrink(self, algorithm, returnsData):
        '''This method is computes the generalised constant-variance-covariance
        matrix and the optimal shrinkage constant as De Nard describes in his paper
        Opps! I Shrunk the Sample Covariance Matrix Again: Blockbuster Meets Shrinkage,
        using a fixed number of groups for the Blockbuster clustering algorithm. It
        returns, the shrinkage constant, the generalised constant-variance-covariance
        matrix and the srhunk sample covariance matrix.
        Arguments:
            algorithm: algorithm object that calls this method.
            returnsData: pandas DataFrame with the past period observations of each
                         asset as rows.
        '''
        
        returnsDataIndex = returnsData.index
        returnsData = returnsData.to_numpy()
    
        # "returnsData" contains the returns of each symbol in rows.
        numAssets, T = returnsData.shape
    
        # Compute centered returns.
        x = (returnsData.T - returnsData.mean(axis=1)).T
    
        # Compute sample covariance, variance and standard deviation matrices.
        samplecovmatrix = np.cov(returnsData, rowvar=True)
        varmatrix = np.diag(np.diag(samplecovmatrix))
        stdmatrix = np.sqrt(varmatrix)
        stdmatrixinv = np.diag(1/np.diag(stdmatrix))
        corrmatrix = np.dot(stdmatrixinv, np.dot(samplecovmatrix, stdmatrixinv))
    
        '''
        Blockbuster Clustering Algorithm.
        '''
        # Compute the eigenvalues and normalised eigenvectors of the sample covariance matrix.
        eigenvals, eigenvects = np.linalg.eig(samplecovmatrix)
        # Select the normalised eigenvectors corresponding to the K highest eigenvalues.
        ind = np.argpartition(eigenvals, -self.AssetGroupCount)[-self.AssetGroupCount:]
        selectedEV = eigenvects[:, ind]
        # Normalise the matrix with the K highest eigenvectors as columns row-wise.
        for i in range(selectedEV.shape[0]):
            selectedEV[i,:] = selectedEV[i,:] / np.linalg.norm(selectedEV[i,:], ord=2)
        # Apply K-means to the rows of the matrix that contains the K selected eigenvectors as rows.
        kmeans = KMeans(n_clusters=self.AssetGroupCount, random_state=100).fit(selectedEV)
        
        # Sort the rows and columns of the sample covariance matrix,
        # so that the block covariance matrix between the assets in the same group lies on the diagonal.
        groupedAssets = pd.DataFrame(index=returnsDataIndex,
                                     data=kmeans.labels_,
                                     columns=["Group"]).sort_values(by="Group")
        # Save the asset clustering for later export.
        AssetClusteringsExport[algorithm.Time] = groupedAssets["Group"]
        pdSampleCovMatrix = pd.DataFrame(data=samplecovmatrix,
                                         index=returnsDataIndex,
                                         columns=returnsDataIndex)
        pdSampleCovMatrix = pdSampleCovMatrix.reindex(index=groupedAssets.index,
                                                      columns=groupedAssets.index)
    
        '''
        Generalised Constant-Variance-Covariance shrinkage target.
        '''
    
        PhiGCVC = []
        for i in range(self.AssetGroupCount):
            ithBlockRow = []
            for j in range(self.AssetGroupCount):
                currentCovMatrixBlock = pdSampleCovMatrix.loc[(groupedAssets.iloc[:,0] == i),
                                                              (groupedAssets.iloc[:,0] == j)]
                if i == j:
                    ijPhiGCVCBlock = np.full(shape=currentCovMatrixBlock.shape,
                                             fill_value=currentCovMatrixBlock.mean())
                    np.fill_diagonal(ijPhiGCVCBlock, np.diagonal(currentCovMatrixBlock).mean())
                    
                else:
                    ijPhiGCVCBlock = np.full(shape=currentCovMatrixBlock.shape,
                                             fill_value=currentCovMatrixBlock.mean())
                
                ithBlockRow.append(ijPhiGCVCBlock)
                
            PhiGCVC.append(ithBlockRow)
            
        PhiGCVC = np.block(PhiGCVC)
        PhiGCVC = pd.DataFrame(data=PhiGCVC,
                               index=pdSampleCovMatrix.index,
                               columns=pdSampleCovMatrix.index)
        PhiGCVC = PhiGCVC.reindex(index=returnsDataIndex,
                                  columns=returnsDataIndex).to_numpy()
    
        '''
        Compute delta shrinkage intensity.
        '''
    
        # Compute r bar.
        rbar = (corrmatrix.sum() - numAssets) / ((numAssets-1) * numAssets)
    
        # Compute pi hat.
        y = np.power(x, 2)
        piMat = np.dot(y, y.T) / T - np.power(samplecovmatrix, 2)
        pihat = piMat.sum()
    
        # Compute rho hat.
        term1 = np.dot(np.power(returnsData, 3), returnsData.T) / T
        term2 = np.dot(varmatrix, samplecovmatrix)
        term3 = np.dot(samplecovmatrix, varmatrix)
        term4 = np.dot(varmatrix, samplecovmatrix)
        thetaMat = term1 - term2 - term3 + term4
        np.fill_diagonal(thetaMat, 0)
        rhohat = np.diag(piMat).sum() + rbar * np.multiply(np.outer(np.diag(stdmatrixinv),
                                                                    np.diag(stdmatrix)),
                                                           thetaMat).sum()
    
        # Compute gamma hat.
        gammahat = np.linalg.norm(PhiGCVC - samplecovmatrix, "fro") ** 2
    
        # Compute shrinkage constant.
        delta = 1/T * (pihat - rhohat) / gammahat
    
        if delta <= 0:
            algorithm.Log("Shrinkage constant is negative or zero.")
            return 0, PhiGCVC, samplecovmatrix
    
        elif delta >= 1:
            algorithm.Log("Shrinkage constant is greater than 1.")
            return 1, PhiGCVC, PhiGCVC
    
        return delta, PhiGCVC, (delta * PhiGCVC + (1-delta) * samplecovmatrix)
        
class HierarchicalRiskParityPortfolio(PortfolioConstructionModel):
    '''PortfolioConstructionModel object that computes the Hierarchical Risk Parity portfolio.
    '''
    
    def __init__(self, lookbackPeriods):
        # Number of past days' returns data to consider for the portfolio construction.
        self.LookbackPeriods = lookbackPeriods
        
    def CreateTargets(self, algorithm, insights):
        '''This method is executed every time new insights are emitted by the
        "AtMonthStartAlphaModel" object. If new insights are emitted, compute
        Hierarchical Risk Parity portfolio for the symbols of the insights.
        Arguments:
            algorithm: algorithm object that calls this method.
            insights: list of insights emitted by the "AtMonthStartAlphaModel" object.
        '''
        
        if len(insights) > 0:
            algorithm.Log("Computing Hierarchical Risk Parity portfolio allocation...")
            
            # Get the past returns of the symbols for which insights were received.
            returnsData = dict()
        
            for symbol in [insight.Symbol for insight in insights]:
                symbolReturns = algorithm.History(symbol,
                                                  self.LookbackPeriods,
                                                  Resolution.Daily).loc[symbol, ["close", "open"]]
                symbolReturns = symbolReturns.apply(lambda x: x.loc["close"] / x.loc["open"] - 1,
                                                    axis=1)
                returnsData[str(symbol)] = symbolReturns
            # Create pandas DataFrame with the symbol names as index sorted alphabetically
            # and the returns of each of the "self.LookbackPeriods" days as columns.
            returnsData = pd.DataFrame.from_dict(returnsData, orient="index").sort_index()
            
            # Compute hierarchical risk parity portfolio.
            # Code from Advances in Financial Machine Learning, Marcos López de Prado.
            # Small changes to the code from pages 240-242 applied to fit QuantConnect's API
            # and the general implementation structure of the program.
            
            # pandas cov() and corr() method compute the covariance and correlation of the columns.
            cov, corr = returnsData.T.cov(), returnsData.T.corr()
        
            # 3) cluster
            # Compute distance matrix based on correlation, where 0 <= d[i,j] <= 1.
            dist = ((1 - corr) / 2.)**.5
            link = sch.linkage(dist, "single")
            # Save the hierarchical structure to later export.
            AssetClusteringsExport[algorithm.Time] = link.reshape(link.size)
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
                
            # Save allocation vector.
            AllocationVectorsExport[algorithm.Time] = w
            
            return [PortfolioTarget.Percent(algorithm, symbol, w[symbol]) for symbol in w.index]
            
        return []

# -----------------------
# Target execution model.
# -----------------------
        
class MarginOptimisedExecutionModel(ExecutionModel):
    '''Custom ExecutionModel class that optimises used margin on rebalancing by first
    rebalancing positions which will be reduced and then rebalancing positions which
    will increased.
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