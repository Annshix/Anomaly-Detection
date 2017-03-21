
sql = "select %d as pattern, A.[DocEntry], A.[CardCode], A.[DocDate], A.[DocDueDate], A.[TaxDate], B.[ItemCode] " \
    "from [YFY_TW].[dbo].[%s] A inner join [YFY_TW].[dbo].[%s] B " \
    "on A.[DocEntry] = B.[DocEntry]"
