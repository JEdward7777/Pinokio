import pinokio2


p = pinokio2.Pinokio2(skip_load=True)
p.selected_pair = pinokio2.SentancePair()
p.selected_pair.output = [1,2,3,4,5,6,7]
p.output = [2,3,1,24,34,6]

print( p._grade_sentance() )
