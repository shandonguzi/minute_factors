from abc import abstractmethod


class AlphaBase:
    # 返回因子集的fields, names
    @abstractmethod
    def get_factors(self):
        pass

    def get_field_by_name(self, name):
        fields, names = self.get_factors()
        for f,n in zip(fields, names):
            if n == name:
                return f


    def get_labels(self):
        return ["shift(close, -5)/shift(open, -1) - 1", "qcut(label_c, 10)"
                ], ["label_c", 'label']

    def get_ic_labels(self):
        days = [1, 5, 10, 20]
        fields = ['shift(close, -{})/close - 1'.format(d) for d in days]
        names = ['return_{}'.format(d) for d in days]
        return fields, names

    def get_all_fields_names(self, b_ic=False):
        fields, names = self.get_factors()
        if not b_ic:
            label_fields, label_names = self.get_labels()
        else:
            label_fields, label_names = self.get_ic_labels()

        fields.extend(label_fields)
        names.extend(label_names)
        return fields, names


class AlphaLit(AlphaBase):
    @staticmethod
    def parse_config_to_fields():
        fields = []
        names = []

        windows = [2, 5, 10, 20]
        fields += ['close/shift(close,%d) - 1' % d for d in windows]
        names += ['roc_%d' % d for d in windows]

        fields += ['avg(volume,1)/avg(volume,5)']
        names += ['avg_amount_1_avg_amount_5']

        fields += ['avg(volume,5)/avg(volume,20)']
        names += ['avg_amount_5_avg_amount_20']

        fields += ['rank(avg(volume,1))/rank(avg(volume,5))']
        names += ['rank_avg_amount_1_avg_amount_5']

        fields += ['avg(volume,5)/avg(volume,20)']
        names += ['rank_avg_amount_5_avg_amount_20']

        windows = [2, 5, 10]
        fields += ['rank(roc_%d)' % d for d in windows]
        names += ['rank_roc_%d' % d for d in windows]

        fields += ['rank(roc_2)/rank(roc_5)']
        names += ['rank_roc_2_rank_roc_5']

        fields += ['rank(roc_5)/rank(roc_10)']
        names += ['rank_roc_5_rank_roc_10']

        return fields, names


class Alpha158(AlphaBase):
    def __init__(self):
        pass

    def get_factors(self):
        # ['CORD30', 'STD30', 'CORR5', 'RESI10', 'CORD60', 'STD5', 'LOW0',
        # 'WVMA30', 'RESI5', 'ROC5', 'KSFT', 'STD20', 'RSV5', 'STD60', 'KLEN']
        fields = []
        names = []

        # kbar
        fields += [
            "(close-open)/open",
            "(high-low)/open",
            "(close-open)/(high-low+1e-12)",
            "(high-greater(open, close))/open",
            "(high-greater(open, close))/(high-low+1e-12)",
            "(less(open, close)-low)/open",
            "(less(open, close)-low)/(high-low+1e-12)",
            "(2*close-high-low)/open",
            "(2*close-high-low)/(high-low+1e-12)",
        ]
        names += [
            "KMID",
            "KLEN",
            "KMID2",
            "KUP",
            "KUP2",
            "KLOW",
            "KLOW2",
            "KSFT",
            "KSFT2",
        ]

        # =========== price ==========
        feature = ["OPEN", "HIGH", "LOW", "CLOSE"]
        windows = range(5)
        for field in feature:
            field = field.lower()
            fields += ["shift(%s, %d)/close" % (field, d) if d != 0 else "%s/close" % field for d in windows]
            names += [field.upper() + str(d) for d in windows]

        # ================ volume ===========
        fields += ["shift(volume, %d)/(volume+1e-12)" % d if d != 0 else "volume/(volume+1e-12)" for d in windows]
        names += ["VOLUME" + str(d) for d in windows]

        # ================= rolling ====================

        windows = [5, 10, 20, 30, 60]
        fields += ["shift(close, %d)/close" % d for d in windows]
        names += ["ROC%d" % d for d in windows]

        fields += ["mean(close, %d)/close" % d for d in windows]
        names += ["MA%d" % d for d in windows]

        fields += ["std(close, %d)/close" % d for d in windows]
        names += ["STD%d" % d for d in windows]

        #fields += ["slope(close, %d)/close" % d for d in windows]
        #names += ["BETA%d" % d for d in windows]

        fields += ["max(high, %d)/close" % d for d in windows]
        names += ["MAX%d" % d for d in windows]

        fields += ["min(low, %d)/close" % d for d in windows]
        names += ["MIN%d" % d for d in windows]

        fields += ["quantile(close, %d, 0.8)/close" % d for d in windows]
        names += ["QTLU%d" % d for d in windows]

        fields += ["quantile(close, %d, 0.2)/close" % d for d in windows]
        names += ["QTLD%d" % d for d in windows]

        #fields += ["ts_rank(close, %d)" % d for d in windows]
        #names += ["RANK%d" % d for d in windows]

        fields += ["(close-min(low, %d))/(max(high, %d)-min(low, %d)+1e-12)" % (d, d, d) for d in windows]
        names += ["RSV%d" % d for d in windows]

        fields += ["idxmax(high, %d)/%d" % (d, d) for d in windows]
        names += ["IMAX%d" % d for d in windows]

        fields += ["idxmin(low, %d)/%d" % (d, d) for d in windows]
        names += ["IMIN%d" % d for d in windows]

        fields += ["(idxmax(high, %d)-idxmin(low, %d))/%d" % (d, d, d) for d in windows]
        names += ["IMXD%d" % d for d in windows]

        fields += ["corr(close, log(volume+1), %d)" % d for d in windows]
        names += ["CORR%d" % d for d in windows]

        fields += ["corr(close/shift(close,1), log(volume/shift(volume, 1)+1), %d)" % d for d in windows]
        names += ["CORD%d" % d for d in windows]

        fields += ["mean(close>shift(close, 1), %d)" % d for d in windows]
        names += ["CNTP%d" % d for d in windows]

        fields += ["mean(close<shift(close, 1), %d)" % d for d in windows]
        names += ["CNTN%d" % d for d in windows]

        fields += ["mean(close>shift(close, 1), %d)-mean(close<shift(close, 1), %d)" % (d, d) for d in windows]
        names += ["CNTD%d" % d for d in windows]

        fields += [
            "sum(greater(close-shift(close, 1), 0), %d)/(sum(Abs(close-shift(close, 1)), %d)+1e-12)" % (d, d)
            for d in windows
        ]
        names += ["SUMP%d" % d for d in windows]

        fields += [
            "sum(greater(shift(close, 1)-close, 0), %d)/(sum(Abs(close-shift(close, 1)), %d)+1e-12)" % (d, d)
            for d in windows
        ]
        names += ["SUMN%d" % d for d in windows]

        fields += [
            "(sum(greater(close-shift(close, 1), 0), %d)-sum(greater(shift(close, 1)-close, 0), %d))"
            "/(sum(Abs(close-shift(close, 1)), %d)+1e-12)" % (d, d, d)
            for d in windows
        ]
        names += ["SUMD%d" % d for d in windows]

        fields += ["mean(volume, %d)/(volume+1e-12)" % d for d in windows]
        names += ["VMA%d" % d for d in windows]

        fields += ["std(volume, %d)/(volume+1e-12)" % d for d in windows]
        names += ["VSTD%d" % d for d in windows]

        fields += [
            "std(Abs(close/shift(close, 1)-1)*volume, %d)/(mean(Abs(close/shift(close, 1)-1)*volume, %d)+1e-12)"
            % (d, d)
            for d in windows
        ]
        names += ["WVMA%d" % d for d in windows]

        fields += [
            "sum(greater(volume-shift(volume, 1), 0), %d)/(sum(Abs(volume-shift(volume, 1)), %d)+1e-12)"
            % (d, d)
            for d in windows
        ]
        names += ["VSUMP%d" % d for d in windows]

        fields += [
            "sum(greater(shift(volume, 1)-volume, 0), %d)/(sum(Abs(volume-shift(volume, 1)), %d)+1e-12)"
            % (d, d)
            for d in windows
        ]
        names += ["VSUMN%d" % d for d in windows]

        fields += [
            "(sum(greater(volume-shift(volume, 1), 0), %d)-sum(greater(shift(volume, 1)-volume, 0), %d))"
            "/(sum(Abs(volume-shift(volume, 1)), %d)+1e-12)" % (d, d, d)
            for d in windows
        ]
        names += ["VSUMD%d" % d for d in windows]

        return fields, names
