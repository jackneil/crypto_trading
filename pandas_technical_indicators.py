"""
Indicators as shown by Peter Bakker at:
https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code

Updated by Jack Neil (jdoc) in 2018 to represent deprecated function calls in pandas 
as well as adding other %'age calculations and some default 'n's based on typical TA usage
https://github.com/jackusc/
"""

# Import Built-Ins
import logging

# Import Third-Party
import pandas as pd

# Import Homebrew

# Init Logging Facilities
log = logging.getLogger(__name__)


def MA(df, n):
    """Moving Average
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['close'].rolling(window=n,center=False).mean(), name = 'MA_' + str(n))
    #MA = pd.Series(pd.rolling_mean(df['close'], n), name = 'MA_' + str(n))
    df = df.join(MA)
    return df

#symlink to MA
def SMA(df, n):
    return MA(df, n)

def EMA(df, n):
    """Exponential Moving Average
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EMA = pd.Series(df['close'].ewm(span=n,min_periods=n-1,adjust=True,ignore_na=False).mean(), 
                    name = 'EMA_' + str(n))
    #EMA = pd.Series(pd.ewma(df['close'], span = n, min_periods = n - 1), name = 'EMA_' + str(n))
    df = df.join(EMA)
    return df


def MOM(df, n=14):
    """Momentum
    
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['close'].diff(n), name = 'Momentum_' + str(n))
    df = df.join(M)
    return df


def ROC(df, n=14):
    """Rate of Change
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['close'].diff(n - 1)
    N = df['close'].shift(n - 1)
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    df = df.join(ROC)
    return df


def ATR(df, n=14):
    """Average True Range
    Returns the average true range over the preceding candle count n at each time,
    along with the percent of close value this range represents

    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span = n, min_periods = n).mean(), name = 'ATR_' + str(n))
    df = df.join(ATR)
    df = df.join(pd.Series(ATR/df['close'], name='ATR_PCT_' + str(n))) 
    return df


def BB(df, n=20):
    """Bollinger Bands
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """   
    MA = pd.Series(df['close'].rolling(window=n).mean())
    MSD = pd.Series(df['close'].rolling(window=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))
    df = df.join(B2)
    return df


def PPSR(df):
    """Pivot Points, Supports and Resistances
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)
    R1 = pd.Series(2 * PP - df['low'])
    S1 = pd.Series(2 * PP - df['high'])
    R2 = pd.Series(PP + df['high'] - df['low'])
    S2 = pd.Series(PP - df['high'] + df['low'])
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


def STOK(df):
    """Stochastic Oscillator %K
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')
    df = df.join(SOk)
    return df


def STOD(df, n=14):
    """Stochastic Oscillator %D

    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')
    SOd = pd.Series(SOk.ewm(span = n, min_periods = n - 1).mean(), name = 'SO%d_' + str(n))
    df = df.join(SOd)
    return df


def TRIX(df, n):
    """Triple Exponentially Smoothed Moving Average (a momentum oscillator)
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EX1 = df['close'].ewm(span = n, min_periods = n - 1).mean()
    EX2 = EX1.ewm(span = n, min_periods = n - 1).mean()
    EX3 = EX2.ewm(span = n, min_periods = n - 1).mean()
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    df = df.join(Trix)
    return df


def ADX(df, n=14, n_ADX=14):
    """Average Directional {Movement} Index
    
    Returns values in the range of 0 <-> 1.0 (multiply * 100 to get into typical range of 0 <-> 100)
    :param df: pandas.DataFrame
    :param n: 
    :param n_ADX: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span = n, min_periods = n).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = n - 1).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = n - 1).mean() / ATR)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span = n_ADX, min_periods = n_ADX - 1).mean(), name = 'ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df


def MACD(df, n_fast=12, n_slow=26):
    """MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['close'].ewm(span = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = pd.Series(df['close'].ewm(span = n_slow, min_periods = n_slow - 1).mean())
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span = 9, min_periods = 8).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def MI(df):
    """Mass Index (predict trend reversals)
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    Range = df['high'] - df['low']
    EX1 = Range.ewm(span = 9, min_periods = 8).mean()
    EX2 = EX1.ewm(span = 9, min_periods = 8).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(window=25).sum(), name = 'Mass Index')
    df = df.join(MassI)
    return df


def VI(df, n):
    """Vortex Indicator
    
    Vortex Indicator described here:
        http://www.vortexindicator.com/VFX_VORTEX.PDF
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.get_value(i + 1, 'high') - df.get_value(i, 'low')) - abs(df.get_value(i + 1, 'low') - df.get_value(i, 'high'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.Series(VM).rolling(window=n).sum() / pd.Series(TR).rolling(window=n).sum(), name = 'Vortex_' + str(n))
    df = df.join(VI)
    return df


def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    """KST Oscillator
    
    :param df: pandas.DataFrame
    :param r1: 
    :param r2: 
    :param r3: 
    :param r4: 
    :param n1: 
    :param n2: 
    :param n3: 
    :param n4: 
    :return: pandas.DataFrame
    """
    M = df['close'].diff(r1 - 1)
    N = df['close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['close'].diff(r2 - 1)
    N = df['close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['close'].diff(r3 - 1)
    N = df['close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['close'].diff(r4 - 1)
    N = df['close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(ROC1.rolling(window=n1).sum() + 
                    ROC2.rolling(window=n2).sum() * 2 + 
                    ROC3.rolling(window=n3).sum() * 3 + 
                    ROC4.rolling(window=n4).sum() * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df


def RSI(df, n=14):
    """Relative Strength Index
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = n - 1, adjust=True, ignore_na=False).mean())
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = n - 1, adjust=True, ignore_na=False).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    df = df.join(RSI)
    return df


def TSI(df, r=25, s=13):
    """True Strength Index
    Output is bound between +100 and -100.  Most values will be between +25 and -25.
    TSI approaching +100 = more overbought ... and vice versa
    Slope of the TSI indicates trend direction (rising = uptrend)
    
    :param df: pandas.DataFrame
    :param r: 
    :param s: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['close'].diff(1))
    aM = abs(M)    
    EMA1 = pd.Series(M.ewm(span = r, min_periods = r - 1).mean())
    aEMA1 = pd.Series(aM.ewm(span = r, min_periods = r - 1).mean())
    EMA2 = pd.Series(EMA1.ewm(span = s, min_periods = s - 1).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span = s, min_periods = s - 1).mean())
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))
    df = df.join(TSI)
    return df


def ACCDIST(df, n):
    """Accumulation/Distribution
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df


def Chaikin(df):
    """Chaikin Oscillator
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    Chaikin = pd.Series(ad.ewm(span = 3, min_periods = 2).mean() - ad.ewm(span = 10, min_periods = 9).mean(), name = 'Chaikin')
    df = df.join(Chaikin)
    return df


def MFI(df, n=14):
    """Money Flow Index and Ratio
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    PP = (df['high'] + df['low'] + df['close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(MFR.rolling(window=n).mean(), name = 'MFI_' + str(n))
    df = df.join(MFI)
    return df


def OBV(df, n):
    """On-Balance volume
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') > 0:
            OBV.append(df.get_value(i + 1, 'volume'))
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') == 0:
            OBV.append(0)
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') < 0:
            OBV.append(-df.get_value(i + 1, 'volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(window=n).mean(), name = 'OBV_' + str(n))
    df = df.join(OBV_ma)
    return df


def FORCE(df, n):
    """Force Index
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    F = pd.Series(df['close'].diff(n) * df['volume'].diff(n), name = 'Force_' + str(n))
    df = df.join(F)
    return df


def EOM(df, n):
    """Ease of Movement
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EoM = (df['high'].diff(1) + df['low'].diff(1)) * (df['high'] - df['low']) / (2 * df['volume'])
    Eom_ma = pd.Series(EoM.rolling(window=n).mean(), name = 'EoM_' + str(n))
    df = df.join(Eom_ma)
    return df


def CCI(df, n):
    """Commodity Channel Index
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    PP = (df['high'] + df['low'] + df['close']) / 3
    CCI = pd.Series((PP - PP.rolling(window=n).mean()) / PP.rolling(window=n).std(), name = 'CCI_' + str(n))
    df = df.join(CCI)
    return df


def COPP(df, n):
    """Coppock Curve
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['close'].diff(int(n * 11 / 10) - 1)
    N = df['close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['close'].diff(int(n * 14 / 10) - 1)
    N = df['close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series((ROC1 + ROC2).ewm(span = n, min_periods = n).mean(), name = 'Copp_' + str(n))
    df = df.join(Copp)
    return df


def KELCH(df, n):
    """Keltner Channel 
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    KelChM = pd.Series(((df['high'] + df['low'] + df['close']) / 3).rolling(window=n).mean(), name = 'KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['high'] - 2 * df['low'] + df['close']) / 3).rolling(window=n).mean(), name = 'KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['high'] + 4 * df['low'] + df['close']) / 3).rolling(window=n).mean(), name = 'KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df


def ULTOSC(df):
    """Ultimate Oscillator
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))
        TR_l.append(TR)
        BP = df.get_value(i + 1, 'close') - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 *pd.Series(BP_l).rolling(window=7).sum() / pd.Series(TR_l).rolling(window=7).sum()) + 
                     (2 * pd.Series(BP_l).rolling(window=14).sum() / pd.Series(TR_l).rolling(window=14).sum()) + 
                     (pd.Series(BP_l).rolling(window=28).sum() / pd.Series(TR_l).rolling(window=28).sum()), name = 'Ultimate_Osc')
    df = df.join(UltO)
    return df


def DONCH(df, n):
    """Donchian Channel

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    dc_l = []
    while i < n - 1:
        dc_l.append(0)
        i += 1

    i = 0
    while i + n - 1 < df.index[-1]:
        dc = max(df['high'].iloc[i:i + n - 1]) - min(df['low'].iloc[i:i + n - 1])
        dc_l.append(dc)
        i += 1

    donchian_chan = pd.Series(dc_l, name='Donchian_' + str(n))
    donchian_chan = donchian_chan.shift(n - 1)
    return df.join(donchian_chan)


def STD(df, n):
    """Standard Deviation
    Returns the standard deviation at each time along with the percent of close value it represents
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    std = pd.Series((df['close']).rolling(window=n).std(), name='STD_' + str(n))
    df = df.join(std)
    df = df.join(pd.Series(std/df['close'], name='STD_PCT_' + str(n)))
    return df
