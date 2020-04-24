import cv2
import numpy as np

# Video Yakalama için VideoCapture kullanıyoruz ve (0) dediğimiz ise kameranın listedeki yeri
#yani eğer 2 tane kameramız varsa 2. kamerayı açmal için (1) yazmamız lazım.
capture = cv2.VideoCapture(0)

# Maskeleyeceğimiz yere 3 değer atıyoruz = History, Threshold, DetectShadows
# History:Arkaplanda kaç frame olduğunda arkaplanı modeli yenileneceğini seçiyor. Yani arkaplanı bir kere daha yeniliyor.Kaç pixel aklında ttucağını seçiyor.
# Threshold: önceki pik frame ve önceki pozisyonu ve sonraki pozisyonu aklında tutuyor. Birbirinde ne kadar farklı olduğunu belirliyor.
#eğer history değerini çok yüksek tutarsak resimdeki değişikliği aklında dah uzun süre tutabilir.
maske = cv2.createBackgroundSubtractorMOG2(500, 750, True)

# hangi frame'de olduğumuzu takip edecek
kare_sayisi = 0

while(1):
	#Anlık frame değerinin return ediyoruz.
	ret, frame = capture.read()

	# Bir değer var mı diye kontrol ediyoruz yoksa çıkıyoruz döngüden
	if not ret:
		break

	kare_sayisi += 1
	#frame'i yeniden boyutlandırıyoruz
	frame_yeni = cv2.resize(frame, (0, 0), fx=1, fy=1) #frame dsize,

	# Yeni frame in boyutunda yuzey maskesi oluşturuyoruz.
	yuzey_maske = maske.apply(frame_yeni)

	# Maske'deki siyah olmayan pixeller sayılıyor.
	pixelsayaci = np.count_nonzero(yuzey_maske)
	#çıktıyı alıyoruz
	print('Görüntü: %d, Değişen pixel sayısı: %d' % (kare_sayisi, pixelsayaci))
	#Maske'nin değerleri vektör olarak x,y,w,h sabitlerine atıyoruz.
	#x,y pencere boyutu ve w,h genişlik ve uzunluğu
	x, y, w, h = cv2.boundingRect(yuzey_maske)

	# Eğer değişen pixel sayısı belli sayıdan yuksekse hareket olarak algılanacak
	# ilk başta frame çok küçük olduğunda siyah ekrala karşılaşabilir ondan dolayı 1'den buyuk olup olmadığını kontrol ediyoruz.
	if (kare_sayisi > 1 and pixelsayaci > 500):
		print('Hareket Algılandı')
		#Hareket edilen yer bulunduğunda dörtgen içine alınıyor
		frame_yeni = cv2.rectangle(frame_yeni, (x, y), (x + w, y + h), (0, 250, 0), 2)
		cv2.putText(frame_yeni, 'Hareket Algilandi', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

	#Pencere oluşturup gösteriyoruz.
	cv2.imshow('Kamera', frame_yeni)
	cv2.imshow('Maske', yuzey_maske)

	k = cv2.waitKey(1) & 0xff
	if k == 27: #esape tuşuna basınca çıkar
		break

#döngüden çıktıktan sonra yakalamalar iptal ediliyor ve tüm pencereleri kapatıyoruz
capture.release()
cv2.destroyAllWindows()